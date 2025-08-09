import pandas as pd
from sqlalchemy import create_engine
import urllib
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error
from feature_engineering import TargetDefiner
from app import Recommender
from database_utils import get_db_engine

def evaluate_price_model(products_df, inventory_df):
    """Loads the acquisition price pipeline and evaluates its MAE on the test set."""
    print("--- 1. Evaluating Acquisition Price Pipeline ---")
    try:
        artifact = joblib.load('models/price_model_pipeline.joblib')
        median_pipeline = artifact['models']['median']
    except FileNotFoundError:
        print("ERROR: `price_model_pipeline.joblib` not found. Please run the trainer first.")
        return

    data = pd.merge(inventory_df, products_df, on='product_id')
    y = data['acquisition_price'].dropna()
    X = data.loc[y.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = median_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"  - Model: Acquisition Price (Median Quantile)")
    print(f"  - Metric: Mean Absolute Error (MAE) on Test Set")
    print(f"  - Result: ${mae:.2f}\n")


def evaluate_velocity_model(products_df, inventory_df, transactions_df):
    """Loads the sales velocity pipeline and prints its classification report on the test set."""
    print("--- 2. Evaluating Sales Velocity Pipeline ---")
    try:
        artifact = joblib.load('models/velocity_model_pipeline.joblib')
        pipeline = artifact['pipeline']
        target_definer = artifact['target_definer']
    except FileNotFoundError:
        print("ERROR: `velocity_model_pipeline.joblib` not found. Please run the trainer first.")
        return

    sold_units = pd.merge(inventory_df[inventory_df['status'] == 'Sold'], transactions_df[['unit_id', 'transaction_date']], on='unit_id')
    data = pd.merge(sold_units, products_df, on='product_id')

    labeled_data = target_definer.transform(data)
    y = labeled_data['mover_category']
    X = labeled_data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"  - Model: Sales Velocity (Classifier)")
    print(f"  - Metric: Classification Report on Test Set")
    print("  - Result:")
    for line in report.split('\n'):
        print(f"    {line}")
    print("")


def evaluate_dynamic_selling_price_model(products_df, inventory_df, transactions_df):
    """Loads the dynamic selling price pipeline and evaluates its MAE on the test set."""
    print("--- 3. Evaluating Dynamic Selling Price Pipeline ---")
    try:
        artifact = joblib.load('models/dynamic_selling_price_pipeline.joblib')
        median_pipeline = artifact['models']['median_fair_market_price']
    except FileNotFoundError:
        print("ERROR: `dynamic_selling_price_pipeline.joblib` not found. Please run the trainer first.")
        return

    sold_units = pd.merge(inventory_df[inventory_df['status'] == 'Sold'], transactions_df[['unit_id', 'transaction_date', 'final_sale_price']], on='unit_id')
    data = pd.merge(sold_units, products_df, on='product_id').dropna(subset=['final_sale_price'])
    
    y = data['final_sale_price']
    X = data
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = median_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"  - Model: Dynamic Selling Price (Fair Market Median)")
    print(f"  - Metric: Mean Absolute Error (MAE) on Test Set")
    print(f"  - Result: ${mae:.2f}\n")


def evaluate_recommender_effectiveness(products_df, transactions_df, compatibility_df):
    """Simulates the recommender's performance against historical purchases. (No changes needed here)"""
    print("--- 4. Evaluating Recommender Effectiveness ---")
    try:
        
        recommender_data = joblib.load('models/recommendation_data.joblib')
        rec_instance = Recommender('models/recommendation_data.joblib', products_df)

    except (FileNotFoundError, ImportError):
        print("ERROR: Could not load `recommendation_data.joblib` or import Recommender class.")
        return

    baskets = transactions_df.groupby('transaction_id')
    phone_accessory_baskets = []
    
    phone_ids_set = set(products_df[products_df['product_type'] == 'Used Phone']['product_id'])
    accessory_ids_set = set(products_df[products_df['product_type'] == 'Accessory']['product_id'])

    for _, basket in baskets:
        if not set(basket['product_id']).isdisjoint(phone_ids_set) and not set(basket['product_id']).isdisjoint(accessory_ids_set):
            phone_accessory_baskets.append(basket)

    if not phone_accessory_baskets:
        print("  - No historical transactions with both a phone and accessory were found. Cannot evaluate.")
        return

    hits, total_opportunities = 0, 0
    for basket in phone_accessory_baskets:
        phone_row = basket[basket['product_id'].isin(phone_ids_set)].iloc[0]
        phone_model_name = products_df[products_df['product_id'] == phone_row['product_id']]['model_name'].iloc[0]
        
        recommended_ids = rec_instance.recommend_accessories(phone_model_name=phone_model_name, top_n=3)
        if not recommended_ids: continue

        purchased_accessory_ids = set(basket[basket['product_id'].isin(accessory_ids_set)]['product_id'])
        if not purchased_accessory_ids.isdisjoint(set(recommended_ids)):
            hits += 1
        total_opportunities += 1

    hit_rate = (hits / total_opportunities) * 100 if total_opportunities > 0 else 0

    print(f"  - Model: Accessory Recommender")
    print(f"  - Metric: Top-3 Hit Rate")
    print(f"  - Result: {hit_rate:.2f}%")
    print(f"  - Interpretation: In {total_opportunities} sales with accessories, the recommender's top 3 suggestions would have included a purchased item {hits} times.\n")

def main():
    print("=========================================")
    print("=      MODEL PERFORMANCE AUDIT          =")
    print("=========================================\n")

    if not os.path.exists('models'):
        print("FATAL: The 'models' directory does not exist. Please run `model_trainer.py` first.")
        return

    db_engine = get_db_engine()
    if db_engine is None: return

    print("--- Loading Data from Database... ---")
    inventory_df = pd.read_sql("SELECT * FROM inventory_units", db_engine)
    products_df = pd.read_sql("SELECT * FROM products", db_engine)
    transactions_df = pd.read_sql("SELECT * FROM transactions", db_engine)
    compat_df = pd.read_sql("SELECT * FROM accessory_compatibility", db_engine)
    
    products_df['base_model'] = products_df['model_name'].str.extract(r'(iPhone \d{1,2}(?: Pro| Plus| Pro Max| Mini)?)')[0]
    
    for df in [inventory_df, transactions_df]:
        for col in ['acquisition_date', 'transaction_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

    print("--- Data Loading Complete. Starting Evaluations... ---\n")

    phones_only_df = products_df[products_df['product_type'] == 'Used Phone']

    evaluate_price_model(phones_only_df, inventory_df)
    evaluate_velocity_model(phones_only_df, inventory_df, transactions_df)
    evaluate_dynamic_selling_price_model(phones_only_df, inventory_df, transactions_df)
    evaluate_recommender_effectiveness(products_df, transactions_df, compat_df)

    print("=========================================")
    print("=          AUDIT COMPLETE               =")
    print("=========================================")
    db_engine.dispose()

if __name__ == '__main__':
    main()