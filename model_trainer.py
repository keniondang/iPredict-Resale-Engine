# -*- coding: utf-8 -*-
"""
A unified script for training all models related to the used phone business,
reading all source data from a SQL Server database.
"""

# 1. IMPORTS
# ==============================================================================
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import urllib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, mean_absolute_error
import lightgbm as lgb
from prophet import Prophet
import joblib
import re
import os
import argparse
from datetime import datetime, timedelta

# 2. DATABASE CONNECTION SETUP
# ==============================================================================

# --- CONFIGURATION ---
# !!! IMPORTANT: UPDATE THESE VALUES TO MATCH YOUR SQL SERVER SETUP !!!
SERVER_NAME = "localhost\\SQLEXPRESS"
DATABASE_NAME = "UsedPhoneResale"
# ---------------------

def get_db_engine():
    """Creates and returns a SQLAlchemy engine for SQL Server."""
    try:
        params = urllib.parse.quote_plus(
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={SERVER_NAME};"
            f"DATABASE={DATABASE_NAME};"
            f"Trusted_Connection=yes;"
        )
        connection_string = f"mssql+pyodbc:///?odbc_connect={params}"
        engine = create_engine(connection_string)
        # Test connection
        connection = engine.connect()
        connection.close()
        print(f"Successfully created DB engine for SQL Server: {SERVER_NAME}, DB: {DATABASE_NAME}")
        return engine
    except Exception as e:
        print(f"FATAL: Could not create database engine. Error: {e}")
        return None

def load_data_from_db(engine, table_name):
    """Loads a table from the database into a pandas DataFrame."""
    try:
        print(f"  - Loading table: '{table_name}'...")
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, engine)
        print(f"    ...Done. Loaded {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error loading table {table_name}: {e}")
        return pd.DataFrame()

# 3. HELPER & UTILITY FUNCTIONS
# ==============================================================================

# Add these two functions to model_trainer.py

def define_target_variable(df):
    """
    Calculates the time an item spent in stock and categorizes it using quantiles.
    """
    df['days_in_stock'] = (df['transaction_date'] - df['acquisition_date']).dt.days
    df = df[df['days_in_stock'] >= 0]

    if len(df['days_in_stock']) < 3: # Handle case with too little data
        return df, None

    fast_threshold = df['days_in_stock'].quantile(0.33)
    medium_threshold = df['days_in_stock'].quantile(0.66)

    print("--- Defining Mover Categories (based on quantiles) ---")
    print(f"Fast Mover: Sold in <= {fast_threshold:.0f} days")
    print(f"Medium Mover: Sold between {fast_threshold:.0f} and {medium_threshold:.0f} days")
    print(f"Dead Stock: Took > {medium_threshold:.0f} days to sell")

    conditions = [
        df['days_in_stock'] <= fast_threshold,
        (df['days_in_stock'] > fast_threshold) & (df['days_in_stock'] <= medium_threshold)
    ]
    choices = ['Fast Mover', 'Medium Mover']
    df['mover_category'] = np.select(conditions, choices, default='Dead Stock')

    return df

def engineer_grade_specific_features(df):
    """
    Creates features based on sales volume and velocity that are specific
    to each product AND each grade.
    """
    df = df.sort_values('transaction_date').copy()
    df_indexed = df.set_index('transaction_date')
    grouping_cols = ['product_id', 'grade']
    
    # Feature 1: Grade-specific sales count in last 120 days
    sales_counts = df_indexed.groupby(grouping_cols)['unit_id'].rolling('120D').count()
    sales_counts = sales_counts.reset_index(name='grade_specific_sales_last_120d')
    df = pd.merge(df, sales_counts, on=['transaction_date', 'product_id', 'grade'], how='left')

    # Feature 2: Grade-specific average days to sell in last 120 days
    avg_days = df_indexed.groupby(grouping_cols)['days_in_stock'].rolling('120D').mean()
    avg_days = avg_days.reset_index(name='grade_specific_avg_days_last_120d')
    df = pd.merge(df, avg_days, on=['transaction_date', 'product_id', 'grade'], how='left')
    
    # Shift features to prevent data leakage
    df['grade_specific_sales_last_120d'] = df.groupby(grouping_cols)['grade_specific_sales_last_120d'].shift(1)
    df['grade_specific_avg_days_last_120d'] = df.groupby(grouping_cols)['grade_specific_avg_days_last_120d'].shift(1)
    
    df.fillna(0, inplace=True)
    return df

def get_predecessor(base_model: str):
    """Finds the predecessor model name (e.g., 'iPhone 14' -> 'iPhone 13')."""
    match = re.search(r'(\d+)', base_model)
    if not match:
        return None
    predecessor_version = int(match.group(1)) - 1
    return base_model.replace(str(match.group(1)), str(predecessor_version))

def _get_market_demand(data_to_score: pd.DataFrame, sales_history: pd.DataFrame, products_ref: pd.DataFrame, date_col: str, primary_lookback: int, predecessor_lookback: int):
    """Generic helper function to calculate the market demand feature, now grade- and storage-specific."""
    print(f"  - Engineering grade- and storage-specific 'market_demand' feature (lookback: {primary_lookback} days)...")

    def calculate_demand(row):
        reference_date = row[date_col]
        base_model = row['base_model']
        grade = row['grade'] 
        storage = row['storage_gb'] # Get storage from the row being scored

        # Filter by storage, grade, and base_model
        demand = sales_history[
            (sales_history['base_model'] == base_model) &
            (sales_history['grade'] == grade) &
            (sales_history['storage_gb'] == storage) & # <-- ADDED THIS LINE
            (sales_history['transaction_date'] < reference_date) &
            (sales_history['transaction_date'] >= reference_date - timedelta(days=primary_lookback))
        ].shape[0]
        
        if demand == 0:
            predecessor = get_predecessor(base_model)
            if predecessor and predecessor in products_ref['base_model'].unique():
                # Filter predecessor by the same storage and grade
                pred_sales = sales_history[
                    (sales_history['base_model'] == predecessor) &
                    (sales_history['grade'] == grade) &
                    (sales_history['storage_gb'] == storage) & # <-- ADDED THIS LINE
                    (sales_history['transaction_date'] < reference_date) &
                    (sales_history['transaction_date'] >= reference_date - timedelta(days=predecessor_lookback))
                ]
                if not pred_sales.empty:
                    # Adjust demand based on the ratio of lookback periods
                    demand = int(np.ceil(pred_sales.shape[0] / (predecessor_lookback / float(primary_lookback))))
        return demand
    return data_to_score.apply(calculate_demand, axis=1)


# 4. MODEL TRAINING & DATA BUILDING FUNCTIONS
# ==============================================================================

def train_price_model(inventory: pd.DataFrame, products: pd.DataFrame, models_dir: str):
    """Trains quantile regression models to predict phone acquisition prices."""
    print("\n--- [Task] Training Buying Price Prediction Model ---")
    print("Step 1: Preparing data for price model...")
    data = pd.merge(inventory, products, on='product_id')
    data['days_since_model_release'] = (data['acquisition_date'] - data['release_date']).dt.days
    data['days_since_successor_release'] = (data['acquisition_date'] - data['successor_release_date']).dt.days
    data['days_since_successor_release'].fillna(-999, inplace=True)
    feature_columns = ['base_model', 'grade', 'storage_gb', 'model_tier', 'original_msrp', 'days_since_model_release', 'days_since_successor_release']
    categorical_features = ['base_model', 'grade', 'model_tier']
    X = data[feature_columns]
    y = data['acquisition_price'].dropna()
    X = X.loc[y.index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)], remainder='passthrough')
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    print("Step 2: Training quantile regression models...")
    models = {}
    quantiles = {'low': 0.2, 'median': 0.5, 'high': 0.8}
    for name, quantile in quantiles.items():
        print(f"  - Training {name} bound model (quantile={quantile})...")
        model = lgb.LGBMRegressor(objective='quantile', alpha=quantile, random_state=42)
        model.fit(X_train_processed, y_train)
        models[name] = model
    print("\nModel Performance:")
    median_model = models['median']
    y_train_pred = median_model.predict(X_train_processed)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    print(f"  - Training MAE (median model): ${train_mae:.2f}")
    print("Step 3: Saving model pipeline...")
    pipeline = {'preprocessor': preprocessor, 'models': models}
    output_path = os.path.join(models_dir, 'price_model.joblib')
    joblib.dump(pipeline, output_path)
    print(f"Price prediction model saved successfully to '{output_path}'")


# Replace the old train_velocity_model function with this one

from sklearn.ensemble import GradientBoostingClassifier # Add this import at the top of the file

def train_velocity_model(inventory: pd.DataFrame, products: pd.DataFrame, transactions: pd.DataFrame, models_dir: str):
    """
    Trains an ADVANCED classification model to predict sales velocity using historical, grade-specific features.
    """
    print("\n--- [Task] Training ADVANCED Sales Velocity Prediction Model ---")

    # Step 1: Merge data to get a comprehensive history of sold phones
    print("Step 1: Preparing data for advanced model...")
    sold_units = pd.merge(inventory[inventory['status'] == 'Sold'], transactions[['unit_id', 'transaction_date']], on='unit_id')
    data = pd.merge(sold_units, products, on='product_id')
    data.dropna(subset=['acquisition_date', 'transaction_date'], inplace=True)

    # Step 2: Define the target variable ('Fast Mover', 'Medium Mover', 'Dead Stock')
    print("Step 2: Defining target variable...")
    labeled_df = define_target_variable(data)

    # Step 3: Engineer sophisticated, grade-specific features
    print("Step 3: Engineering grade-specific historical features...")
    featured_df = engineer_grade_specific_features(labeled_df)

    # Step 4: Train the Gradient Boosting model
    print("Step 4: Training Gradient Boosting Classifier...")
    features = [
        'model_tier', 'storage_gb', 'original_msrp',
        'grade_specific_sales_last_120d',
        'grade_specific_avg_days_last_120d'
    ]
    target = 'mover_category'

    model_df = featured_df.dropna(subset=features + [target])
    if model_df.empty:
        print("Could not train velocity model due to lack of historical data.")
        return

    X = model_df[features]
    y = model_df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Using the new GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("\nModel Performance on Test Set:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    # Step 5: Save the model AND the data required for prediction
    print("Step 5: Saving model and prediction data artifacts...")
    
    # The API will now need the model, the feature list, and the historical data with features
    pipeline = {
        'model': model,
        'features': features,
        'historical_data': featured_df
    }
    output_path = os.path.join(models_dir, 'velocity_model_advanced.joblib')
    joblib.dump(pipeline, output_path)
    print(f"Advanced sales velocity pipeline saved successfully to '{output_path}'")


def train_dynamic_selling_price_model(inventory: pd.DataFrame, products: pd.DataFrame, transactions: pd.DataFrame, models_dir: str):
    """Trains dynamic quantile regression models for setting the selling price."""
    print("\n--- [Task] Training Dynamic Selling Price Prediction Model ---")
    print("Step 1: Preparing data for dynamic price model...")
    sold_units = pd.merge(inventory[inventory['status'] == 'Sold'], transactions[['unit_id', 'transaction_date', 'final_sale_price']], on='unit_id')
    data = pd.merge(sold_units, products, on='product_id')
    data.dropna(subset=['final_sale_price', 'acquisition_date', 'transaction_date'], inplace=True)
    print("Step 2: Engineering features from the perspective of the sale date...")
    data['days_since_model_release'] = (data['transaction_date'] - data['release_date']).dt.days
    data['days_since_successor_release'] = (data['transaction_date'] - data['successor_release_date']).dt.days
    data['days_since_successor_release'].fillna(-999, inplace=True)
    
    # This sales_history is for a different model and does not need the grade-specific logic
    sales_history = pd.merge(transactions, products, on='product_id')[['base_model', 'transaction_date']]
    
    # Local helper function for this model, which is NOT grade-specific
    def _get_demand_for_selling_price(data_to_score, sales_hist, prods_ref, date_col):
        def calc_demand(row):
            ref_date = row[date_col]
            base_mod = row['base_model']
            demand = sales_hist[(sales_hist['base_model'] == base_mod) & (sales_hist['transaction_date'] < ref_date) & (sales_hist['transaction_date'] >= ref_date - timedelta(days=7))].shape[0]
            if demand == 0:
                pred = get_predecessor(base_mod)
                if pred and pred in prods_ref['base_model'].unique():
                    demand = sales_hist[(sales_hist['base_model'] == pred) & (sales_hist['transaction_date'] < ref_date) & (sales_hist['transaction_date'] >= ref_date - timedelta(days=7))].shape[0]
            return demand
        return data_to_score.apply(calc_demand, axis=1)

    data['market_demand_7_days'] = _get_demand_for_selling_price(data, sales_history, products, 'transaction_date')
    
    feature_columns = [
        'original_msrp', 
        'storage_gb', 
        'grade', 
        'model_tier', 
        'base_model', 
        'days_since_model_release', 
        'days_since_successor_release', 
        'market_demand_7_days'
    ]
    
    categorical_features = ['grade', 'model_tier', 'base_model']
    X = data[feature_columns]
    y = data['final_sale_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Step 3: Training quantile regression models with early stopping...")
    preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)], remainder='passthrough')
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    models = {}
    quantiles = {'low_liquidate_price': 0.2, 'median_fair_market_price': 0.5, 'high_start_price': 0.8}
    for name, quantile in quantiles.items():
        print(f"  - Training '{name}' model (quantile={quantile})...")
        model = lgb.LGBMRegressor(objective='quantile', alpha=quantile, random_state=42, learning_rate=0.05, n_estimators=1000, num_leaves=31, reg_lambda=0.1)
        model.fit(X_train_processed, y_train, eval_set=[(X_test_processed, y_test)], eval_metric='quantile', callbacks=[lgb.early_stopping(50, verbose=False)])
        models[name] = model
    print("\nModel Performance:")
    median_model = models['median_fair_market_price']
    y_train_pred = median_model.predict(X_train_processed)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    print(f"  - Training MAE (median model): ${train_mae:.2f}")
    print("Step 4: Saving model pipeline...")
    pipeline = {'preprocessor': preprocessor, 'models': models}
    output_path = os.path.join(models_dir, 'dynamic_selling_price_pipeline.joblib')
    joblib.dump(pipeline, output_path)
    print(f"Dynamic selling price model saved successfully to '{output_path}'")


def train_forecast_model(category_value: str, products_df: pd.DataFrame, transactions_df: pd.DataFrame, models_dir: str):
    """Trains a Prophet time series model for a specific product."""
    print(f"\n--- Training Demand Forecast Model for: {category_value} ---")
    model_info_rows = products_df[products_df['model_name'] == category_value]
    if model_info_rows.empty:
        print(f"Skipping '{category_value}': Could not find product info.")
        return
    model_info = model_info_rows.iloc[0]
    release_date = model_info['release_date']
    successor_date = model_info['successor_release_date']
    sales_data = pd.merge(transactions_df, products_df, on='product_id')
    sales_after_release = sales_data[sales_data['transaction_date'] >= release_date]
    category_sales = sales_after_release[sales_after_release['model_name'] == category_value].copy()
    if len(category_sales) < 20:
        print(f"Skipping '{category_value}': Not enough sales data to train a reliable model.")
        return
    daily_sales = category_sales.groupby('transaction_date').agg(sales_count=('transaction_id', 'count')).reset_index()
    daily_sales.rename(columns={'transaction_date': 'ds', 'sales_count': 'y'}, inplace=True)
    print("  - Engineering dynamic 'cap' for product lifecycle...")
    max_sales = daily_sales['y'].max()
    base_cap = max_sales * 1.5
    def calculate_cap(row_date):
        days_since_release = (row_date - release_date).days
        decay_factor = 1 - (days_since_release / (365 * 2))
        if pd.notna(successor_date) and row_date > successor_date:
            days_after_successor = (row_date - successor_date).days
            successor_decay = 0.5 - (days_after_successor / (365 * 1))
            decay_factor = min(decay_factor, successor_decay)
        return max(1, base_cap * decay_factor)
    daily_sales['cap'] = daily_sales['ds'].apply(calculate_cap)
    daily_sales['floor'] = 0
    print("  - Training Prophet model with 'logistic' growth...")
    model = Prophet(growth='logistic', daily_seasonality=False)
    model.fit(daily_sales)
    forecast = model.predict(daily_sales[['ds', 'cap', 'floor']])
    mae = mean_absolute_error(daily_sales['y'], forecast['yhat'])
    print(f"  - In-Sample (Training) MAE: {mae:.2f} units/day")
    model_package = {'model': model, 'release_date': release_date, 'successor_date': successor_date, 'base_cap': base_cap}
    safe_category_value = "".join(c for c in category_value if c.isalnum())
    output_path = os.path.join(models_dir, f'demand_forecast_{safe_category_value}.joblib')
    joblib.dump(model_package, output_path)
    print(f"Model for '{category_value}' saved successfully.")


# FILE: model_trainer.py

def build_discontinuation_list(products: pd.DataFrame, transactions: pd.DataFrame, compatibility: pd.DataFrame, accessory_inventory: pd.DataFrame, models_dir: str, sales_threshold: int = 10, days_to_check: int = 30):
    """Analyzes sales data to identify accessories for obsolete phones."""
    print("\n--- [Task] Building Discontinuation Alert List ---")
    print("Step 1: Preparing data for analysis...")
    phones_df = products[products['product_type'] == 'Used Phone']
    cutoff_date = datetime.now() - timedelta(days=days_to_check)
    recent_phone_sales = transactions[(transactions['product_id'].isin(phones_df['product_id'])) & (transactions['transaction_date'] >= cutoff_date)]
    accessories_in_stock = accessory_inventory[accessory_inventory['quantity'] > 0]
    discontinuation_list = []
    print(f"Step 2: Analyzing {len(accessories_in_stock['product_id'].unique())} unique in-stock accessories...")
    for acc_id in accessories_in_stock['product_id'].unique():
        compatible_phones = compatibility[compatibility['accessory_product_id'] == acc_id]
        if compatible_phones.empty:
            continue
        compatible_phone_ids = compatible_phones['phone_product_id'].tolist()
        total_sales = recent_phone_sales[recent_phone_sales['product_id'].isin(compatible_phone_ids)].shape[0]
        if total_sales < sales_threshold:
            accessory_info = products[products['product_id'] == acc_id].iloc[0]
            
            # FIX: Cast NumPy integer types to standard Python integers here
            discontinuation_list.append({
                'product_id': int(acc_id),
                'model_name': accessory_info['model_name'],
                'compatible_phone_sales_past_180_days': int(total_sales)
            })

    print(f"Step 3: Found {len(discontinuation_list)} accessories to recommend for discontinuation.")
    output_path = os.path.join(models_dir, 'discontinuation_list.joblib')
    joblib.dump(discontinuation_list, output_path)
    print(f"Discontinuation list saved successfully to '{output_path}'.")

def build_recommendation_data(products_df: pd.DataFrame, transactions_df: pd.DataFrame, compatibility_df: pd.DataFrame, models_dir: str):
    """Pre-calculates data for the hybrid recommendation engine."""
    print("\n--- [Task] Building Recommendation Data ---")
    
    # Part 1: Calculate Overall Popularity
    print("Step 1: Calculating overall accessory popularity...")
    accessories = products_df[products_df['product_type'] == 'Accessory']
    accessory_sales = transactions_df[transactions_df['product_id'].isin(accessories['product_id'])]
    accessory_popularity = accessory_sales['product_id'].value_counts()

    # Part 2: Calculate Co-purchase Counts (Market Basket Analysis)
    print("Step 2: Performing market basket analysis for co-purchases...")
    transactions_with_products = pd.merge(transactions_df, products_df[['product_id', 'product_type', 'base_model']], on='product_id', how='left')
    baskets = transactions_with_products.groupby('transaction_id')
    co_purchase_counts = {}
    for _, basket in baskets:
        phone_in_basket = basket[basket['product_type'] == 'Used Phone']
        accessories_in_basket = basket[basket['product_type'] == 'Accessory']
        if not phone_in_basket.empty and not accessories_in_basket.empty:
            phone_base_model = phone_in_basket['base_model'].iloc[0]
            for acc_id in accessories_in_basket['product_id']:
                key = (phone_base_model, acc_id)
                co_purchase_counts[key] = co_purchase_counts.get(key, 0) + 1

    co_purchase_df = pd.DataFrame(list(co_purchase_counts.items()), columns=['key', 'count'])
    co_purchase_df[['phone_base_model', 'accessory_product_id']] = pd.DataFrame(co_purchase_df['key'].tolist(), index=co_purchase_df.index)
    co_purchase_df.drop(columns=['key'], inplace=True)

    # Part 3: Package and Save
    print("Step 3: Packaging and saving all recommendation data...")
    recommendation_package = {'popularity': accessory_popularity, 'compatibility': compatibility_df, 'co_purchase': co_purchase_df}
    output_path = os.path.join(models_dir, 'recommendation_data.joblib')
    joblib.dump(recommendation_package, output_path)
    print(f"\nRecommendation data built and saved successfully to '{output_path}'.")


# 5. MAIN EXECUTION BLOCK
# ==============================================================================
def main():
    """Main function to parse arguments and run selected training tasks."""
    parser = argparse.ArgumentParser(description="Unified Model Training Script for the Used Phone Business.")
    parser.add_argument('--all', action='store_true', help="Run all training and build tasks.")
    parser.add_argument('--price', action='store_true', help="Train the acquisition price model.")
    parser.add_argument('--velocity', action='store_true', help="Train the sales velocity model.")
    parser.add_argument('--dynamic-price', action='store_true', help="Train the dynamic selling price model.")
    parser.add_argument('--forecast', action='store_true', help="Train demand forecast models for all products.")
    parser.add_argument('--discontinuation', action='store_true', help="Build the accessory discontinuation list.")
    parser.add_argument('--recommendation', action='store_true', help="Build the recommendation data artifact.")
    args = parser.parse_args()

    MODELS_DIR = 'models'
    os.makedirs(MODELS_DIR, exist_ok=True)
    run_all = args.all or not any(vars(args).values())

    db_engine = get_db_engine()
    if db_engine is None:
        return # Stop execution if DB connection fails

    print("\n--- Loading and Preparing All Data Sources from Database ---")
    try:
        inventory_df = load_data_from_db(db_engine, 'inventory_units')
        products_df = load_data_from_db(db_engine, 'products')
        transactions_df = load_data_from_db(db_engine, 'transactions')
        if run_all or args.discontinuation or args.recommendation:
            compat_df = load_data_from_db(db_engine, 'accessory_compatibility')
        if run_all or args.discontinuation:
            acc_inv_df = load_data_from_db(db_engine, 'accessory_inventory')
    except Exception as e:
        print(f"Error loading data from database: {e}.")
        return

    # Convert date columns to datetime objects
    for df in [inventory_df, products_df, transactions_df]:
        for col in ['acquisition_date', 'release_date', 'successor_release_date', 'transaction_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

    # Engineer 'base_model' feature
    products_df['base_model'] = products_df['model_name'].str.extract(r'(iPhone \d{1,2}(?: Pro| Plus| Pro Max| Mini)?)')[0]
    print("--- Data Loading and Preparation Complete ---")

    if run_all or args.price:
        train_price_model(inventory_df, products_df[products_df['product_type'] == 'Used Phone'], MODELS_DIR)
    if run_all or args.velocity:
        train_velocity_model(inventory_df, products_df[products_df['product_type'] == 'Used Phone'], transactions_df, MODELS_DIR)
    if run_all or args.dynamic_price:
        train_dynamic_selling_price_model(inventory_df, products_df[products_df['product_type'] == 'Used Phone'], transactions_df, MODELS_DIR)
    if run_all or args.forecast:
        print("\n--- [Task] Starting Batch Training for All Specific iPhone Products ---")
        phone_products = products_df[products_df['product_type'] == 'Used Phone']
        unique_products = phone_products['model_name'].dropna().unique()
        print(f"Found {len(unique_products)} unique iPhone products to train.")
        for product_name in unique_products:
            train_forecast_model(product_name, phone_products, transactions_df, MODELS_DIR)
        print("\n--- Batch forecast training complete. ---")
    if run_all or args.discontinuation:
        build_discontinuation_list(products_df, transactions_df, compat_df, acc_inv_df, MODELS_DIR)
    if run_all or args.recommendation:
        build_recommendation_data(products_df, transactions_df, compat_df, MODELS_DIR)
        
    print("\n--- All selected tasks are complete. ---")
    db_engine.dispose()
    print("Database engine disposed.")

if __name__ == '__main__':
    main()