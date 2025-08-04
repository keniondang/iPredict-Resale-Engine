import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import re
import os
from datetime import datetime, timedelta

def _get_market_demand(data_to_score: pd.DataFrame, sales_history: pd.DataFrame, products_ref: pd.DataFrame):
    """
    Helper function to calculate market demand based on sales in the 7 days
    prior to the transaction date. Includes cold-start logic.
    """
    print("  - Engineering 'market_demand_7_days' feature (this may take a moment)...")
    
    def get_predecessor(base_model: str):
        match = re.search(r'(\d+)', base_model)
        if not match: return None
        return base_model.replace(str(match.group(1)), str(int(match.group(1)) - 1))

    def calculate_demand(row):
        transaction_date = row['transaction_date']
        base_model = row['base_model']
        
        # Direct demand for the same base model in the last 7 days
        demand = sales_history[
            (sales_history['base_model'] == base_model) &
            (sales_history['transaction_date'] < transaction_date) &
            (sales_history['transaction_date'] >= transaction_date - timedelta(days=7))
        ].shape[0]

        # Cold-start logic: If no direct demand, check predecessor's 7-day sales
        if demand == 0:
            predecessor = get_predecessor(base_model)
            if predecessor and predecessor in products_ref['base_model'].unique():
                pred_sales = sales_history[
                    (sales_history['base_model'] == predecessor) &
                    (sales_history['transaction_date'] < transaction_date) &
                    (sales_history['transaction_date'] >= transaction_date - timedelta(days=7))
                ]
                demand = pred_sales.shape[0]
        return demand

    return data_to_score.apply(calculate_demand, axis=1)

def train_dynamic_selling_price_model():
    """
    Loads data, trains the dynamic quantile regression models for selling price,
    and saves the final pipeline.
    """
    print("--- Training Dynamic Selling Price Prediction Model ---")

    # 1. Load Data
    print("Step 1: Loading data from /data folder...")
    try:
        inventory = pd.read_csv('data/inventory_units.csv')
        products = pd.read_csv('data/products.csv')
        transactions = pd.read_csv('data/transactions.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure all CSV files are in the 'data/' directory.")
        return

    # 2. Prepare Data
    print("Step 2: Preparing data and engineering features...")
    products = products[products['product_type'] == 'Used Phone'].copy()
    for df in [inventory, products, transactions]:
        for col in ['acquisition_date', 'release_date', 'successor_release_date', 'transaction_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
    products['base_model'] = products['model_name'].str.extract(r'(iPhone \d{1,2}(?: Pro| Plus| Pro Max| Mini)?)')[0]

    # Create the training dataset by merging all data for sold units
    sold_units = pd.merge(inventory[inventory['status'] == 'Sold'], transactions[['unit_id', 'transaction_date', 'final_sale_price']], on='unit_id')
    data = pd.merge(sold_units, products, on='product_id')
    data.dropna(subset=['final_sale_price', 'acquisition_date', 'transaction_date'], inplace=True)

    # 3. Feature Engineering
    print("Step 3: Engineering features from the perspective of the sale date...")
    data['days_in_inventory'] = (data['transaction_date'] - data['acquisition_date']).dt.days
    data['days_since_model_release'] = (data['transaction_date'] - data['release_date']).dt.days
    data['days_since_successor_release'] = (data['transaction_date'] - data['successor_release_date']).dt.days
    data['days_since_successor_release'].fillna(-999, inplace=True)
    
    # Create a clean sales history reference for demand calculation
    sales_history = data[['base_model', 'transaction_date']].copy()
    data['market_demand_7_days'] = _get_market_demand(data, sales_history, products)

    # 4. Define Features, Target, and Split
    feature_columns = [
        'original_msrp', 'storage_gb', 'grade', 'model_tier', 'base_model', 
        'acquisition_price', 'days_in_inventory', 'days_since_model_release', 
        'days_since_successor_release', 'market_demand_7_days'
    ]
    categorical_features = ['grade', 'model_tier', 'base_model']
    
    X = data[feature_columns]
    y = data['final_sale_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Create and Train Pipeline
    print("Step 4: Training quantile regression models with tuned hyperparameters...")
    preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)], remainder='passthrough')
    preprocessor.fit(X_train)
    
    # Create processed validation set for early stopping
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    models = {}
    quantiles = {
        'low_liquidate_price': 0.2, 
        'median_fair_market_price': 0.5, 
        'high_start_price': 0.8
    }
    for name, quantile in quantiles.items():
        print(f"  - Training '{name}' model (quantile={quantile})...")
        # Tuned hyperparameters for more robust predictions
        model = lgb.LGBMRegressor(
            objective='quantile', 
            alpha=quantile, 
            random_state=42,
            learning_rate=0.05,
            n_estimators=1000, # Increased estimators to work with lower learning rate
            num_leaves=31,
            reg_lambda=0.1 # L2 regularization
        )
        
        # Use early stopping to prevent overfitting
        model.fit(
            X_train_processed, y_train,
            eval_set=[(X_test_processed, y_test)],
            eval_metric='quantile',
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        models[name] = model
    
    # 6. Save Pipeline
    print("Step 5: Saving model pipeline...")
    pipeline = {'preprocessor': preprocessor, 'models': models}
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/dynamic_selling_price_pipeline.joblib')
    
    print("\nDynamic selling price model trained and saved successfully to 'models/dynamic_selling_price_pipeline.joblib'")

if __name__ == "__main__":
    train_dynamic_selling_price_model()