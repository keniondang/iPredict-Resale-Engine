import pandas as pd
from sqlalchemy import create_engine
import urllib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, mean_absolute_error
import lightgbm as lgb
from prophet import Prophet
import joblib
import os
import argparse
from datetime import datetime, timedelta
from feature_engineering import (
    DateFeatureCalculator,
    MarketDemandCalculator,
    GradeHistoryCalculator,
    TargetDefiner
)

SERVER_NAME = "localhost\\SQLEXPRESS"
DATABASE_NAME = "UsedPhoneResale"

def get_db_engine():
    try:
        params = urllib.parse.quote_plus(f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER_NAME};DATABASE={DATABASE_NAME};Trusted_Connection=yes;")
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
        connection = engine.connect()
        connection.close()
        print(f"Successfully created DB engine for SQL Server: {SERVER_NAME}, DB: {DATABASE_NAME}")
        return engine
    except Exception as e:
        print(f"FATAL: Could not create database engine. Error: {e}")
        return None

def load_data_from_db(engine, table_name):
    try:
        print(f"  - Loading table: '{table_name}'...")
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, engine)
        print(f"    ...Done. Loaded {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error loading table {table_name}: {e}")
        return pd.DataFrame()

def train_price_model(inventory: pd.DataFrame, products: pd.DataFrame, models_dir: str):
    print("\n--- [Task] Training Buying Price Prediction Model ---")
    
    data = pd.merge(inventory, products, on='product_id')
    
    categorical_features = ['base_model', 'grade', 'model_tier', 'month_of_year', 'is_holiday_season']
    numerical_features = ['storage_gb', 'original_msrp', 'days_since_model_release', 'days_since_successor_release']
    
    y = data['acquisition_price'].dropna()
    X = data.loc[y.index]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('numerical', 'passthrough', numerical_features)
        ],
        remainder='drop'
    )
    
    models = {}
    for name, quantile in {'low': 0.2, 'median': 0.5, 'high': 0.8}.items():
        print(f"  - Training {name} bound model (quantile={quantile})...")
        pipeline = Pipeline(steps=[
            ('feature_calculator', DateFeatureCalculator(reference_date_col='acquisition_date')),
            ('preprocessor', preprocessor),
            ('regressor', lgb.LGBMRegressor(objective='quantile', alpha=quantile, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        models[name] = pipeline

    print("\nModel Performance:")
    train_mae = mean_absolute_error(y_train, models['median'].predict(X_train))
    print(f"  - Training MAE (median model): ${train_mae:.2f}")

    joblib.dump({'models': models}, os.path.join(models_dir, 'price_model_pipeline.joblib'))
    print(f"Price prediction pipeline saved successfully.")

def train_velocity_model(inventory: pd.DataFrame, products: pd.DataFrame, transactions: pd.DataFrame, models_dir: str):
    print("\n--- [Task] Training ADVANCED Sales Velocity Prediction Model ---")

    sold_units = pd.merge(inventory[inventory['status'] == 'Sold'], transactions[['unit_id', 'transaction_date']], on='unit_id')
    data = pd.merge(sold_units, products, on='product_id')
    
    target_definer = TargetDefiner().fit(data)
    labeled_data = target_definer.transform(data)
    
    grade_history_calc = GradeHistoryCalculator().fit(labeled_data)

    y = labeled_data['mover_category']
    X = labeled_data
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    final_model_features = [
        'model_tier', 'storage_gb', 'original_msrp',
        'grade_specific_sales_last_120d',
        'grade_specific_avg_days_last_120d'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('selector', 'passthrough', final_model_features)
        ],
        remainder='drop'
    )

    pipeline = Pipeline(steps=[
        ('history_features', grade_history_calc),
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)

    print("\nModel Performance on Test Set:")
    print(classification_report(y_test, pipeline.predict(X_test), zero_division=0))

    final_artifact = {'pipeline': pipeline, 'target_definer': target_definer}
    joblib.dump(final_artifact, os.path.join(models_dir, 'velocity_model_pipeline.joblib'))
    print(f"Advanced sales velocity pipeline saved successfully.")

def train_dynamic_selling_price_model(inventory: pd.DataFrame, products: pd.DataFrame, transactions: pd.DataFrame, models_dir: str):
    print("\n--- [Task] Training Dynamic Selling Price Prediction Model ---")
    
    sold_units = pd.merge(inventory[inventory['status'] == 'Sold'], transactions[['unit_id', 'transaction_date', 'final_sale_price']], on='unit_id')
    data = pd.merge(sold_units, products, on='product_id').dropna(subset=['final_sale_price'])
    
    sales_history = pd.merge(transactions, products, on='product_id')
    market_demand_calc = MarketDemandCalculator(lookback_days=7).fit(sales_history)
    
    y = data['final_sale_price']
    X = data
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_features = ['grade', 'model_tier', 'base_model', 'month_of_year', 'is_holiday_season']
    numerical_features = ['storage_gb', 'original_msrp', 'days_since_model_release', 'days_since_successor_release', 'market_demand']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('numerical', 'passthrough', numerical_features)
        ],
        remainder='drop'
    )
    
    models = {}
    for name, quantile in {'low_liquidate_price': 0.2, 'median_fair_market_price': 0.5, 'high_start_price': 0.8}.items():
        print(f"  - Training '{name}' model (quantile={quantile})...")
        pipeline = Pipeline(steps=[
            ('date_calculator', DateFeatureCalculator(reference_date_col='transaction_date')),
            ('demand_calculator', market_demand_calc),
            ('preprocessor', preprocessor),
            ('regressor', lgb.LGBMRegressor(objective='quantile', alpha=quantile, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        models[name] = pipeline

    print("\nModel Performance:")
    train_mae = mean_absolute_error(y_train, models['median_fair_market_price'].predict(X_train))
    print(f"  - Training MAE (median model): ${train_mae:.2f}")

    joblib.dump({'models': models}, os.path.join(models_dir, 'dynamic_selling_price_pipeline.joblib'))
    print(f"Dynamic selling price pipeline saved successfully.")

def train_forecast_model(category_value: str, products_df: pd.DataFrame, transactions_df: pd.DataFrame, models_dir: str):
    print(f"\n--- Training Demand Forecast Model for: {category_value} ---")
    model_info_rows = products_df[products_df['model_name'] == category_value]
    if model_info_rows.empty: return
    model_info = model_info_rows.iloc[0]
    release_date, successor_date = model_info['release_date'], model_info['successor_release_date']
    sales_data = pd.merge(transactions_df, products_df, on='product_id')
    category_sales = sales_data[(sales_data['transaction_date'] >= release_date) & (sales_data['model_name'] == category_value)].copy()
    if len(category_sales) < 20: return
    daily_sales = category_sales.groupby('transaction_date').size().reset_index(name='y')
    daily_sales.rename(columns={'transaction_date': 'ds'}, inplace=True)
    max_sales, base_cap = daily_sales['y'].max(), daily_sales['y'].max() * 1.5
    def calculate_cap(row_date):
        decay_factor = 1 - ((row_date - release_date).days / (365 * 2))
        if pd.notna(successor_date) and row_date > successor_date:
            successor_decay = 0.5 - ((row_date - successor_date).days / 365)
            decay_factor = min(decay_factor, successor_decay)
        return max(0.01, base_cap * decay_factor)
    daily_sales['cap'], daily_sales['floor'] = daily_sales['ds'].apply(calculate_cap), 0
    model = Prophet(growth='logistic', daily_seasonality=False).fit(daily_sales)
    safe_name = "".join(c for c in category_value if c.isalnum())
    joblib.dump({'model': model, 'release_date': release_date, 'successor_date': successor_date, 'base_cap': base_cap}, os.path.join(models_dir, f'demand_forecast_{safe_name}.joblib'))
    print(f"Model for '{category_value}' saved successfully.")

def build_discontinuation_list(products: pd.DataFrame, transactions: pd.DataFrame, compatibility: pd.DataFrame, accessory_inventory: pd.DataFrame, models_dir: str, sales_threshold: int = 10, days_to_check: int = 30):
    print("\n--- [Task] Building Discontinuation Alert List ---")
    cutoff = datetime.now() - timedelta(days=days_to_check)
    recent_sales = transactions[(transactions['product_id'].isin(products[products['product_type'] == 'Used Phone']['product_id'])) & (transactions['transaction_date'] >= cutoff)]
    in_stock_acc = accessory_inventory[accessory_inventory['quantity'] > 0]
    discontinuation_list = []
    for acc_id in in_stock_acc['product_id'].unique():
        phone_ids = compatibility[compatibility['accessory_product_id'] == acc_id]['phone_product_id'].tolist()
        if not phone_ids: continue
        if recent_sales[recent_sales['product_id'].isin(phone_ids)].shape[0] < sales_threshold:
            info = products[products['product_id'] == acc_id].iloc[0]
            discontinuation_list.append({'product_id': int(acc_id), 'model_name': info['model_name'], 'compatible_phone_sales_past_180_days': int(recent_sales[recent_sales['product_id'].isin(phone_ids)].shape[0])})
    joblib.dump(discontinuation_list, os.path.join(models_dir, 'discontinuation_list.joblib'))
    print(f"Discontinuation list saved successfully.")

def build_recommendation_data(products_df: pd.DataFrame, transactions_df: pd.DataFrame, compatibility_df: pd.DataFrame, models_dir: str):
    print("\n--- [Task] Building Recommendation Data ---")
    acc_sales = transactions_df[transactions_df['product_id'].isin(products_df[products_df['product_type'] == 'Accessory']['product_id'])]
    popularity = acc_sales['product_id'].value_counts()
    merged = pd.merge(transactions_df, products_df[['product_id', 'product_type', 'base_model']], on='product_id')
    baskets = merged.groupby('transaction_id')
    counts = {}
    for _, basket in baskets:
        phone = basket[basket['product_type'] == 'Used Phone']
        accessories = basket[basket['product_type'] == 'Accessory']
        if not phone.empty and not accessories.empty:
            base_model = phone['base_model'].iloc[0]
            for acc_id in accessories['product_id']:
                counts[(base_model, acc_id)] = counts.get((base_model, acc_id), 0) + 1
    co_purchase = pd.DataFrame(list(counts.items()), columns=['key', 'count'])
    co_purchase[['phone_base_model', 'accessory_product_id']] = pd.DataFrame(co_purchase['key'].tolist(), index=co_purchase.index)
    joblib.dump({'popularity': popularity, 'compatibility': compatibility_df, 'co_purchase': co_purchase.drop(columns=['key'])}, os.path.join(models_dir, 'recommendation_data.joblib'))
    print(f"Recommendation data built and saved successfully.")

def main():
    parser = argparse.ArgumentParser(description="Unified Model Training Script")
    parser.add_argument('--all', action='store_true', help="Run all tasks.")
    parser.add_argument('--price', action='store_true')
    parser.add_argument('--velocity', action='store_true')
    parser.add_argument('--dynamic-price', action='store_true')
    parser.add_argument('--forecast', action='store_true')
    parser.add_argument('--discontinuation', action='store_true')
    parser.add_argument('--recommendation', action='store_true')
    args = parser.parse_args()

    MODELS_DIR = 'models'
    os.makedirs(MODELS_DIR, exist_ok=True)
    run_all = args.all or not any(vars(args).values())

    db_engine = get_db_engine()
    if db_engine is None: return

    print("\n--- Loading Data ---")
    inventory_df = load_data_from_db(db_engine, 'inventory_units')
    products_df = load_data_from_db(db_engine, 'products')
    transactions_df = load_data_from_db(db_engine, 'transactions')
    compat_df = load_data_from_db(db_engine, 'accessory_compatibility')
    acc_inv_df = load_data_from_db(db_engine, 'accessory_inventory')

    for df in [inventory_df, products_df, transactions_df]:
        for col in ['acquisition_date', 'release_date', 'successor_release_date', 'transaction_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

    products_df['base_model'] = products_df['model_name'].str.extract(r'(iPhone \d{1,2}(?: Pro| Plus| Pro Max| Mini)?)')[0]
    
    phones_only_df = products_df[products_df['product_type'] == 'Used Phone']

    if run_all or args.price:
        train_price_model(inventory_df, phones_only_df, MODELS_DIR)
    if run_all or args.velocity:
        train_velocity_model(inventory_df, phones_only_df, transactions_df, MODELS_DIR)
    if run_all or args.dynamic_price:
        train_dynamic_selling_price_model(inventory_df, phones_only_df, transactions_df, MODELS_DIR)
    if run_all or args.forecast:
        for name in phones_only_df['model_name'].dropna().unique():
            train_forecast_model(name, phones_only_df, transactions_df, MODELS_DIR)
    if run_all or args.discontinuation:
        build_discontinuation_list(products_df, transactions_df, compat_df, acc_inv_df, MODELS_DIR)
    if run_all or args.recommendation:
        build_recommendation_data(products_df, transactions_df, compat_df, MODELS_DIR)
        
    print("\n--- All selected tasks are complete. ---")
    db_engine.dispose()

if __name__ == '__main__':
    main()