import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import re
import os
from datetime import datetime, timedelta

def _get_market_demand(data_to_score: pd.DataFrame, sales_history: pd.DataFrame, products_ref: pd.DataFrame):
    """Helper function to calculate the market demand feature."""
    print("  - Engineering 'market_demand' feature (this may take a moment)...")
    
    def get_predecessor(base_model: str):
        match = re.search(r'(\d+)', base_model)
        if not match: return None
        return base_model.replace(str(match.group(1)), str(int(match.group(1)) - 1))

    def calculate_demand(row):
        acquisition_date = row['acquisition_date']
        base_model = row['base_model']
        
        # Direct demand
        demand = sales_history[
            (sales_history['base_model'] == base_model) &
            (sales_history['transaction_date'] < acquisition_date) &
            (sales_history['transaction_date'] >= acquisition_date - timedelta(days=14))
        ].shape[0]

        # Cold-start logic
        if demand == 0:
            predecessor = get_predecessor(base_model)
            if predecessor and predecessor in products_ref['base_model'].unique():
                pred_sales = sales_history[
                    (sales_history['base_model'] == predecessor) &
                    (sales_history['transaction_date'] < acquisition_date) &
                    (sales_history['transaction_date'] >= acquisition_date - timedelta(days=90))
                ]
                if not pred_sales.empty:
                    demand = int(np.ceil(pred_sales.shape[0] / (90 / 7.0)))
        return demand

    return data_to_score.apply(calculate_demand, axis=1)

def train_velocity_model():
    """
    Loads data, trains the classification model for sales velocity,
    and saves the trained pipeline.
    """
    print("--- Training Sales Velocity Prediction Model ---")

    # 1. Load Data
    print("Step 1: Loading data...")
    try:
        inventory = pd.read_csv('data/inventory_units.csv')
        products = pd.read_csv('data/products.csv')
        transactions = pd.read_csv('data/transactions.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure all CSV files are present.")
        return

    # 2. Prepare Data
    print("Step 2: Preparing data and engineering features...")
    products = products[products['product_type'] == 'Used Phone'].copy()
    for df in [inventory, products, transactions]:
        for col in ['acquisition_date', 'release_date', 'transaction_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
    products['base_model'] = products['model_name'].str.extract(r'(iPhone \d{1,2}(?: Pro| Plus| Pro Max| Mini)?)')[0]

    # Create the training dataset (only sold units)
    sold_units = pd.merge(inventory[inventory['status'] == 'Sold'], transactions[['unit_id', 'transaction_date']], on='unit_id')
    data = pd.merge(sold_units, products, on='product_id')

    # Engineer target variable
    data['days_to_sell'] = (data['transaction_date'] - data['acquisition_date']).dt.days
    data['sales_velocity_category'] = pd.cut(data['days_to_sell'], bins=[-1, 30, 90, np.inf], labels=['Fast Mover', 'Medium Mover', 'Dead Stock Risk'])
    
    # FIX: Drop rows where the target variable could not be calculated, which causes the NaN error.
    original_rows = len(data)
    data.dropna(subset=['sales_velocity_category'], inplace=True)
    if original_rows > len(data):
        print(f"  - Dropped {original_rows - len(data)} rows with invalid target values (e.g., missing dates).")

    # Engineer features
    data['days_since_model_release'] = (data['acquisition_date'] - data['release_date']).dt.days
    sales_history = data[['base_model', 'transaction_date']].copy()
    data['market_demand'] = _get_market_demand(data, sales_history, products)

    # 3. Define Features, Target, and Split
    feature_columns = ['grade', 'storage_gb', 'model_tier', 'original_msrp', 'days_since_model_release', 'market_demand']
    categorical_features = ['grade', 'model_tier']
    X = data[feature_columns]
    y = data['sales_velocity_category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # 4. Create and Train Pipeline
    print("Step 3: Training classification model...")
    preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)], remainder='passthrough')
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    pipeline.fit(X_train, y_train)

    # 5. Evaluate and Save
    print("\nModel Performance:")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    print("Step 4: Saving model pipeline...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/velocity_model.joblib')
    print("\nSales velocity model trained and saved successfully to 'models/velocity_model.joblib'")

if __name__ == "__main__":
    train_velocity_model()
