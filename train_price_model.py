import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

def train_price_model():
    """
    Loads data, trains the quantile regression models for price prediction,
    and saves the trained pipeline.
    """
    print("--- Training Buying Price Prediction Model ---")

    # 1. Load Data
    print("Step 1: Loading data...")
    try:
        inventory = pd.read_csv('data/inventory_units.csv')
        products = pd.read_csv('data/products.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure inventory_units.csv and products.csv are present.")
        return

    # 2. Prepare Data
    print("Step 2: Preparing data and engineering features...")
    # Filter for phones and convert date columns
    products = products[products['product_type'] == 'Used Phone'].copy()
    for df in [inventory, products]:
        for col in ['acquisition_date', 'release_date', 'successor_release_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

    # Derive base_model from model_name
    products['base_model'] = products['model_name'].str.extract(r'(iPhone \d{1,2}(?: Pro| Plus| Pro Max| Mini)?)')[0]

    # Merge data and create features
    data = pd.merge(inventory, products, on='product_id')
    data['days_since_model_release'] = (data['acquisition_date'] - data['release_date']).dt.days
    data['days_since_successor_release'] = (data['acquisition_date'] - data['successor_release_date']).dt.days
    data['days_since_successor_release'].fillna(-999, inplace=True)

    # 3. Define Features and Target
    feature_columns = [
        'base_model', 'grade', 'storage_gb', 'model_tier', 'original_msrp', 
        'days_since_model_release', 'days_since_successor_release'
    ]
    categorical_features = ['base_model', 'grade', 'model_tier']
    
    X = data[feature_columns]
    y = data['acquisition_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Create Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
        remainder='passthrough'
    )
    preprocessor.fit(X_train)

    # 5. Train Models
    print("Step 3: Training quantile regression models...")
    X_train_processed = preprocessor.transform(X_train)
    
    models = {}
    quantiles = {'low': 0.2, 'median': 0.5, 'high': 0.8}
    for name, quantile in quantiles.items():
        print(f"  - Training {name} bound model (quantile={quantile})...")
        model = lgb.LGBMRegressor(objective='quantile', alpha=quantile, random_state=42)
        model.fit(X_train_processed, y_train)
        models[name] = model
    
    # 6. Save Pipeline
    print("Step 4: Saving model pipeline...")
    pipeline = {'preprocessor': preprocessor, 'models': models}
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/price_model.joblib')
    
    print("\nPrice prediction model trained and saved successfully to 'models/price_model.joblib'")

if __name__ == "__main__":
    train_price_model()
