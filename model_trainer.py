# -*- coding: utf-8 -*-
"""
A unified script for training all models related to the used phone business,
including purchase price prediction, sales velocity, dynamic selling price,
demand forecasting, discontinuation lists, and recommendation data.
"""

# 1. IMPORTS
# ==============================================================================
import pandas as pd
import numpy as np
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

# 2. HELPER & UTILITY FUNCTIONS
# ==============================================================================

def get_predecessor(base_model: str):
    """Finds the predecessor model name (e.g., 'iPhone 14' -> 'iPhone 13')."""
    match = re.search(r'(\d+)', base_model)
    if not match:
        return None
    predecessor_version = int(match.group(1)) - 1
    return base_model.replace(str(match.group(1)), str(predecessor_version))

def _get_market_demand(data_to_score: pd.DataFrame, sales_history: pd.DataFrame, products_ref: pd.DataFrame, date_col: str, primary_lookback: int, predecessor_lookback: int):
    """Generic helper function to calculate the market demand feature."""
    print(f"  - Engineering 'market_demand' feature (lookback: {primary_lookback} days)...")

    def calculate_demand(row):
        reference_date = row[date_col]
        base_model = row['base_model']
        demand = sales_history[
            (sales_history['base_model'] == base_model) &
            (sales_history['transaction_date'] < reference_date) &
            (sales_history['transaction_date'] >= reference_date - timedelta(days=primary_lookback))
        ].shape[0]
        if demand == 0:
            predecessor = get_predecessor(base_model)
            if predecessor and predecessor in products_ref['base_model'].unique():
                pred_sales = sales_history[
                    (sales_history['base_model'] == predecessor) &
                    (sales_history['transaction_date'] < reference_date) &
                    (sales_history['transaction_date'] >= reference_date - timedelta(days=predecessor_lookback))
                ]
                if not pred_sales.empty:
                    demand = int(np.ceil(pred_sales.shape[0] / (predecessor_lookback / float(primary_lookback))))
        return demand
    return data_to_score.apply(calculate_demand, axis=1)


# 3. MODEL TRAINING & DATA BUILDING FUNCTIONS
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


def train_velocity_model(inventory: pd.DataFrame, products: pd.DataFrame, transactions: pd.DataFrame, models_dir: str):
    """Trains a classification model to predict sales velocity."""
    print("\n--- [Task] Training Sales Velocity Prediction Model ---")
    print("Step 1: Preparing data and engineering features...")
    sold_units = pd.merge(inventory[inventory['status'] == 'Sold'], transactions[['unit_id', 'transaction_date']], on='unit_id')
    data = pd.merge(sold_units, products, on='product_id')
    data['days_to_sell'] = (data['transaction_date'] - data['acquisition_date']).dt.days
    data['sales_velocity_category'] = pd.cut(data['days_to_sell'], bins=[-1, 30, 90, np.inf], labels=['Fast Mover', 'Medium Mover', 'Dead Stock Risk'])
    data.dropna(subset=['sales_velocity_category'], inplace=True)
    data['days_since_model_release'] = (data['acquisition_date'] - data['release_date']).dt.days
    sales_history = pd.merge(transactions, products, on='product_id')[['base_model', 'transaction_date']]
    data['market_demand'] = _get_market_demand(data, sales_history, products, 'acquisition_date', 14, 90)
    feature_columns = ['grade', 'storage_gb', 'model_tier', 'original_msrp', 'days_since_model_release', 'market_demand']
    categorical_features = ['grade', 'model_tier']
    X = data[feature_columns]
    y = data['sales_velocity_category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print("Step 2: Training classification model...")
    preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)], remainder='passthrough')
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))])
    pipeline.fit(X_train, y_train)
    print("\nModel Performance:")
    print("--- Training Set ---")
    y_train_pred = pipeline.predict(X_train)
    print(classification_report(y_train, y_train_pred))
    print("--- Test Set ---")
    y_test_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_test_pred))
    print("Step 3: Saving model pipeline...")
    output_path = os.path.join(models_dir, 'velocity_model.joblib')
    joblib.dump(pipeline, output_path)
    print(f"Sales velocity model saved successfully to '{output_path}'")


def train_dynamic_selling_price_model(inventory: pd.DataFrame, products: pd.DataFrame, transactions: pd.DataFrame, models_dir: str):
    """Trains dynamic quantile regression models for setting the selling price."""
    print("\n--- [Task] Training Dynamic Selling Price Prediction Model ---")
    print("Step 1: Preparing data for dynamic price model...")
    sold_units = pd.merge(inventory[inventory['status'] == 'Sold'], transactions[['unit_id', 'transaction_date', 'final_sale_price']], on='unit_id')
    data = pd.merge(sold_units, products, on='product_id')
    data.dropna(subset=['final_sale_price', 'acquisition_date', 'transaction_date'], inplace=True)
    print("Step 2: Engineering features from the perspective of the sale date...")
    data['days_in_inventory'] = (data['transaction_date'] - data['acquisition_date']).dt.days
    data['days_since_model_release'] = (data['transaction_date'] - data['release_date']).dt.days
    data['days_since_successor_release'] = (data['transaction_date'] - data['successor_release_date']).dt.days
    data['days_since_successor_release'].fillna(-999, inplace=True)
    sales_history = pd.merge(transactions, products, on='product_id')[['base_model', 'transaction_date']]
    data['market_demand_7_days'] = _get_market_demand(data, sales_history, products, 'transaction_date', 7, 7)
    feature_columns = ['original_msrp', 'storage_gb', 'grade', 'model_tier', 'base_model', 'acquisition_price', 'days_in_inventory', 'days_since_model_release', 'days_since_successor_release', 'market_demand_7_days']
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
        decay_factor = 1 - (days_since_release / (365 * 4))
        if pd.notna(successor_date) and row_date > successor_date:
            days_after_successor = (row_date - successor_date).days
            successor_decay = 0.5 - (days_after_successor / (365 * 2))
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


def build_discontinuation_list(products: pd.DataFrame, transactions: pd.DataFrame, compatibility: pd.DataFrame, accessory_inventory: pd.DataFrame, models_dir: str, sales_threshold: int = 10, days_to_check: int = 180):
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
            discontinuation_list.append({'product_id': acc_id, 'model_name': accessory_info['model_name'], 'compatible_phone_sales_past_180_days': total_sales})
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


# 4. MAIN EXECUTION BLOCK
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

    DATA_DIR = 'data'
    MODELS_DIR = 'models'
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("--- Loading and Preparing All Data Sources ---")
    run_all = args.all or not any(vars(args).values())

    try:
        inventory_df = pd.read_csv(os.path.join(DATA_DIR, 'inventory_units.csv'))
        products_df = pd.read_csv(os.path.join(DATA_DIR, 'products.csv'))
        transactions_df = pd.read_csv(os.path.join(DATA_DIR, 'transactions.csv'))
        if run_all or args.discontinuation or args.recommendation:
            compat_df = pd.read_csv(os.path.join(DATA_DIR, 'accessory_compatibility.csv'))
        if run_all or args.discontinuation:
            acc_inv_df = pd.read_csv(os.path.join(DATA_DIR, 'accessory_inventory.csv'))
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure all required CSV files are in the '{DATA_DIR}/' directory.")
        return

    for df in [inventory_df, products_df, transactions_df]:
        for col in ['acquisition_date', 'release_date', 'successor_release_date', 'transaction_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
    products_df['base_model'] = products_df['model_name'].str.extract(r'(iPhone \d{1,2}(?: Pro| Plus| Pro Max| Mini)?)')[0]
    print("--- Data Loading Complete ---")

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

if __name__ == '__main__':
    main()