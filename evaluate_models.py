# -*- coding: utf-8 -*-
"""
Model Evaluation Script

This script loads the pre-trained models and evaluates their performance
on the test portion of the dataset. It is designed to be run after
`model_trainer.py` has successfully created the models.
"""

# 1. IMPORTS
# ==============================================================================
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import urllib
import joblib
import os
import re
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error

# Import necessary functions from the trainer script to ensure data is processed identically
from model_trainer import get_db_engine, define_target_variable, engineer_grade_specific_features

# 2. EVALUATION FUNCTIONS
# ==============================================================================

def evaluate_price_model(db_engine, products_df, inventory_df):
    """Loads the acquisition price model and evaluates its MAE on the test set."""
    print("--- 1. Evaluating Acquisition Price Model ---")
    try:
        pipeline = joblib.load('models/price_model.joblib')
        preprocessor = pipeline['preprocessor']
        median_model = pipeline['models']['median']
    except FileNotFoundError:
        print("ERROR: `price_model.joblib` not found. Please run the trainer first.")
        return

    # --- Data Preparation (must be identical to training) ---
    data = pd.merge(inventory_df, products_df, on='product_id')
    data['acquisition_date'] = pd.to_datetime(data['acquisition_date'])
    data['release_date'] = pd.to_datetime(data['release_date'])
    data['successor_release_date'] = pd.to_datetime(data['successor_release_date'])
    
    data['days_since_model_release'] = (data['acquisition_date'] - data['release_date']).dt.days
    data['days_since_successor_release'] = (data['acquisition_date'] - data['successor_release_date']).dt.days
    data['days_since_successor_release'] = data['days_since_successor_release'].fillna(-999)

    # --- FIX: Add the missing seasonality features ---
    data['month_of_year'] = data['acquisition_date'].dt.month
    data['is_holiday_season'] = data['month_of_year'].isin([11, 12]).astype(int)
    
    feature_columns = [
        'base_model', 'grade', 'storage_gb', 'model_tier', 
        'original_msrp', 'days_since_model_release', 'days_since_successor_release',
        'month_of_year', 'is_holiday_season'
    ]
    
    X = data[feature_columns]
    y = data['acquisition_price'].dropna()
    X = X.loc[y.index]

    # --- Recreate the exact same train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Evaluation ---
    X_test_processed = preprocessor.transform(X_test)
    y_pred = median_model.predict(X_test_processed)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"  - Model: Acquisition Price (Median Quantile)")
    print(f"  - Metric: Mean Absolute Error (MAE)")
    print(f"  - Result: ${mae:.2f}")
    print("  - Interpretation: On average, the model's prediction for the acquisition price is off by this amount.\n")


def evaluate_velocity_model(db_engine, products_df, inventory_df, transactions_df):
    """Loads the sales velocity model and prints its classification report."""
    print("--- 2. Evaluating Sales Velocity Model ---")
    try:
        pipeline = joblib.load('models/velocity_model_advanced.joblib')
        model = pipeline['model']
    except FileNotFoundError:
        print("ERROR: `velocity_model_advanced.joblib` not found. Please run the trainer first.")
        return

    # --- Data Preparation (must be identical to training) ---
    sold_units = pd.merge(inventory_df[inventory_df['status'] == 'Sold'], transactions_df[['unit_id', 'transaction_date']], on='unit_id')
    data = pd.merge(sold_units, products_df, on='product_id')
    data['acquisition_date'] = pd.to_datetime(data['acquisition_date'])
    data['transaction_date'] = pd.to_datetime(data['transaction_date'])
    data.dropna(subset=['acquisition_date', 'transaction_date'], inplace=True)

    labeled_df = define_target_variable(data)
    featured_df = engineer_grade_specific_features(labeled_df)

    features = [
        'model_tier', 'storage_gb', 'original_msrp',
        'grade_specific_sales_last_120d',
        'grade_specific_avg_days_last_120d'
    ]
    target = 'mover_category'

    model_df = featured_df.dropna(subset=features + [target])
    if model_df.empty:
        print("Could not evaluate velocity model due to lack of historical data.")
        return

    X = model_df[features]
    y = model_df[target]

    # --- Recreate the exact same train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- Evaluation ---
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"  - Model: Sales Velocity (Classifier)")
    print(f"  - Metric: Classification Report")
    print("  - Result:")
    for line in report.split('\n'):
        print(f"    {line}")
    print("  - Interpretation: The F1-score is the harmonic mean of precision and recall. Higher is better.")
    print("    It shows how well the model identifies each category ('Fast Mover', 'Dead Stock', etc.).\n")


def evaluate_dynamic_selling_price_model(db_engine, products_df, inventory_df, transactions_df):
    """Loads the dynamic selling price model and evaluates its MAE on the test set."""
    print("--- 3. Evaluating Dynamic Selling Price Model ---")
    try:
        pipeline = joblib.load('models/dynamic_selling_price_pipeline.joblib')
        preprocessor = pipeline['preprocessor']
        median_model = pipeline['models']['median_fair_market_price']
    except FileNotFoundError:
        print("ERROR: `dynamic_selling_price_pipeline.joblib` not found. Please run the trainer first.")
        return

    # --- Data Preparation (must be identical to training) ---
    sold_units = pd.merge(inventory_df[inventory_df['status'] == 'Sold'], transactions_df[['unit_id', 'transaction_date', 'final_sale_price']], on='unit_id')
    data = pd.merge(sold_units, products_df, on='product_id')
    data.dropna(subset=['final_sale_price', 'acquisition_date', 'transaction_date'], inplace=True)
    
    data['transaction_date'] = pd.to_datetime(data['transaction_date'])
    data['release_date'] = pd.to_datetime(data['release_date'])
    data['successor_release_date'] = pd.to_datetime(data['successor_release_date'])

    data['days_since_model_release'] = (data['transaction_date'] - data['release_date']).dt.days
    data['days_since_successor_release'] = (data['transaction_date'] - data['successor_release_date']).dt.days
    data['days_since_successor_release'] = data['days_since_successor_release'].fillna(-999)
    
    # --- FIX: Add the missing seasonality features ---
    data['month_of_year'] = data['transaction_date'].dt.month
    data['is_holiday_season'] = data['month_of_year'].isin([11, 12]).astype(int)

    sales_history = pd.merge(transactions_df, products_df, on='product_id')[['base_model', 'transaction_date']]
    sales_history['transaction_date'] = pd.to_datetime(sales_history['transaction_date'])

    def _get_demand_for_selling_price(data_to_score, sales_hist, prods_ref, date_col):
        def get_predecessor(base_model: str):
            match = re.search(r'(\d+)', base_model)
            if not match: return None
            return base_model.replace(str(match.group(1)), str(int(match.group(1)) - 1))
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

    data['market_demand_7_days'] = _get_demand_for_selling_price(data, sales_history, products_df, 'transaction_date')
    
    feature_columns = [
        'original_msrp', 'storage_gb', 'grade', 'model_tier', 
        'base_model', 'days_since_model_release', 'days_since_successor_release', 
        'market_demand_7_days', 'month_of_year', 'is_holiday_season'
    ]
    
    X = data[feature_columns]
    y = data['final_sale_price']

    # --- Recreate the exact same train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Evaluation ---
    X_test_processed = preprocessor.transform(X_test)
    y_pred = median_model.predict(X_test_processed)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"  - Model: Dynamic Selling Price (Fair Market Median)")
    print(f"  - Metric: Mean Absolute Error (MAE)")
    print(f"  - Result: ${mae:.2f}")
    print("  - Interpretation: On average, the model's prediction for the final selling price is off by this amount.\n")


def evaluate_recommender_effectiveness(db_engine, products_df, transactions_df, compatibility_df):
    """Simulates the recommender's performance against historical purchases."""
    print("--- 4. Evaluating Recommender Effectiveness ---")
    try:
        from app import Recommender
        recommender = Recommender('models/recommendation_data.joblib', products_df)
    except (FileNotFoundError, ImportError):
        print("ERROR: Could not load `recommendation_data.joblib` or import Recommender class.")
        return

    baskets = transactions_df.groupby('transaction_id')
    phone_accessory_baskets = []
    for _, basket in baskets:
        has_phone = any(basket['product_id'].isin(products_df[products_df['product_type'] == 'Used Phone']['product_id']))
        has_accessory = any(basket['product_id'].isin(products_df[products_df['product_type'] == 'Accessory']['product_id']))
        if has_phone and has_accessory:
            phone_accessory_baskets.append(basket)

    if not phone_accessory_baskets:
        print("  - No historical transactions with both a phone and accessory were found. Cannot evaluate.")
        return

    hits = 0
    total_opportunities = 0

    for basket in phone_accessory_baskets:
        phone_row = basket[basket['product_id'].isin(products_df[products_df['product_type'] == 'Used Phone']['product_id'])].iloc[0]
        phone_model_name = products_df[products_df['product_id'] == phone_row['product_id']]['model_name'].iloc[0]
        
        recommended_ids = recommender.recommend_accessories(phone_model_name=phone_model_name, top_n=3)
        
        if not recommended_ids:
            continue

        purchased_accessory_ids = set(basket[basket['product_id'].isin(products_df[products_df['product_type'] == 'Accessory']['product_id'])]['product_id'])
        
        if not purchased_accessory_ids.isdisjoint(set(recommended_ids)):
            hits += 1
        
        total_opportunities += 1

    hit_rate = (hits / total_opportunities) * 100 if total_opportunities > 0 else 0

    print(f"  - Model: Accessory Recommender")
    print(f"  - Metric: Top-3 Hit Rate")
    print(f"  - Result: {hit_rate:.2f}%")
    print(f"  - Interpretation: In {total_opportunities} historical sales where a customer bought a phone and accessory,")
    print(f"    the recommender's top 3 suggestions would have included an item they purchased {hits} times.\n")


# 3. MAIN EXECUTION BLOCK
# ==============================================================================
def main():
    """Main function to run all evaluation tasks."""
    print("=========================================")
    print("=      MODEL PERFORMANCE AUDIT          =")
    print("=========================================\n")

    if not os.path.exists('models'):
        print("FATAL: The 'models' directory does not exist. Please run `model_trainer.py` first.")
        return

    db_engine = get_db_engine()
    if db_engine is None:
        return

    print("--- Loading Data from Database... ---")
    try:
        inventory_df = pd.read_sql("SELECT * FROM inventory_units", db_engine)
        products_df = pd.read_sql("SELECT * FROM products", db_engine)
        transactions_df = pd.read_sql("SELECT * FROM transactions", db_engine)
        compat_df = pd.read_sql("SELECT * FROM accessory_compatibility", db_engine)
        
        products_df['base_model'] = products_df['model_name'].str.extract(r'(iPhone \d{1,2}(?: Pro| Plus| Pro Max| Mini)?)')[0]
        
        print("--- Data Loading Complete. Starting Evaluations... ---\n")
    except Exception as e:
        print(f"FATAL: Error loading data from database: {e}.")
        return

    evaluate_price_model(db_engine, products_df[products_df['product_type'] == 'Used Phone'], inventory_df)
    evaluate_velocity_model(db_engine, products_df[products_df['product_type'] == 'Used Phone'], inventory_df, transactions_df)
    evaluate_dynamic_selling_price_model(db_engine, products_df[products_df['product_type'] == 'Used Phone'], inventory_df, transactions_df)
    evaluate_recommender_effectiveness(db_engine, products_df, transactions_df, compat_df)

    print("=========================================")
    print("=          AUDIT COMPLETE               =")
    print("=========================================")
    db_engine.dispose()

if __name__ == '__main__':
    main()
