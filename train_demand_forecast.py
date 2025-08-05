import pandas as pd
from prophet import Prophet
import os
import joblib
from datetime import timedelta

def train_forecast_model(category_col: str, category_value: str, products_df: pd.DataFrame, transactions_df: pd.DataFrame):
    """
    Trains a Prophet time series model with a logistic growth trend
    for a specific product (e.g., 'iPhone 14 Pro 128GB'), starting from its release date.

    Args:
        category_col (str): The column to filter by (now 'model_name').
        category_value (str): The specific product name to filter for.
        products_df (pd.DataFrame): The pre-loaded products dataframe.
        transactions_df (pd.DataFrame): The pre-loaded transactions dataframe.
    """
    print(f"\n--- Training Demand Forecast Model for: {category_value} ---")

    # 1. Define file paths and model parameters
    output_folder = 'models'
    # Sanitize the filename to handle spaces and special characters
    safe_category_value = "".join(c for c in category_value if c.isalnum())
    output_path = os.path.join(output_folder, f'demand_forecast_{safe_category_value}.joblib')

    # 2. Get model-specific information
    # The category_col is now 'model_name', so we filter by it
    model_info_rows = products_df[products_df[category_col] == category_value]
    if model_info_rows.empty:
        print(f"Skipping '{category_value}': Could not find product info.")
        return
    model_info = model_info_rows.iloc[0]
    
    release_date = pd.to_datetime(model_info['release_date'])
    successor_date = pd.to_datetime(model_info['successor_release_date'])

    # 3. Merge and prepare data, starting from the release date
    sales_data = pd.merge(transactions_df, products_df, on='product_id')
    sales_data['transaction_date'] = pd.to_datetime(sales_data['transaction_date'])
    
    sales_after_release = sales_data[sales_data['transaction_date'] >= release_date]
    category_sales = sales_after_release[sales_after_release[category_col] == category_value].copy()

    if category_sales.empty or len(category_sales) < 20:
        print(f"Skipping '{category_value}': Not enough sales data after its release date to train a reliable model.")
        return

    # 4. Aggregate data for Prophet
    daily_sales = category_sales.groupby('transaction_date').agg(
        sales_count=('transaction_id', 'count')
    ).reset_index()
    daily_sales.rename(columns={'transaction_date': 'ds', 'sales_count': 'y'}, inplace=True)

    # 5. Engineer Dynamic Cap for Logistic Growth
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
    
    # 6. Train the Prophet model with logistic growth
    print("  - Training Prophet model with 'logistic' growth...")
    model = Prophet(growth='logistic', daily_seasonality=False) 
    model.fit(daily_sales)

    # 7. Save the trained model and its parameters
    model_package = {
        'model': model,
        'release_date': release_date,
        'successor_date': successor_date,
        'base_cap': base_cap
    }
    os.makedirs(output_folder, exist_ok=True)
    joblib.dump(model_package, output_path)

    print(f"Model for '{category_value}' trained and saved successfully.")

if __name__ == "__main__":
    print("--- Starting Batch Training for All Specific iPhone Products ---")
    data_folder = 'data'
    products_path = os.path.join(data_folder, 'products.csv')
    transactions_path = os.path.join(data_folder, 'transactions.csv')
    
    try:
        products_df = pd.read_csv(products_path)
        transactions_df = pd.read_csv(transactions_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure all CSV files are in the '{data_folder}/' directory.")
        exit()

    # This column is still needed for the successor date logic inside the function
    products_df['base_model'] = products_df['model_name'].str.extract(r'(iPhone \d{1,2}(?: Pro| Plus| Pro Max| Mini)?)')[0]
    
    # --- CHANGE: Iterate through specific model_name instead of base_model ---
    unique_products = products_df[products_df['product_type'] == 'Used Phone']['model_name'].dropna().unique()

    print(f"Found {len(unique_products)} unique iPhone products to train.")

    for product_name in unique_products:
        train_forecast_model(
            category_col='model_name', 
            category_value=product_name,
            products_df=products_df,
            transactions_df=transactions_df
        )
    
    print("\n--- Batch training complete. ---")
