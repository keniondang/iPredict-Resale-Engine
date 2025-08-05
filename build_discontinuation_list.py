import pandas as pd
import os
import joblib
from datetime import datetime, timedelta

def build_discontinuation_list(sales_threshold: int = 10, days_to_check: int = 180):
    """
    Analyzes inventory and sales data to identify accessories for obsolete
    phones and saves the list to a file.

    Args:
        sales_threshold (int): The minimum number of compatible phone sales required
                               in the period to keep an accessory active.
        days_to_check (int): The lookback period in days for checking phone sales.
    """
    print("--- Building Discontinuation Alert List ---")

    # 1. Define file paths
    data_folder = 'data'
    products_path = os.path.join(data_folder, 'products.csv')
    transactions_path = os.path.join(data_folder, 'transactions.csv')
    compatibility_path = os.path.join(data_folder, 'accessory_compatibility.csv')
    inventory_path = os.path.join(data_folder, 'accessory_inventory.csv')
    output_folder = 'models'
    output_path = os.path.join(output_folder, 'discontinuation_list.joblib')

    # 2. Load all necessary data
    print("Step 1: Loading data...")
    try:
        products_df = pd.read_csv(products_path)
        transactions_df = pd.read_csv(transactions_path)
        compatibility_df = pd.read_csv(compatibility_path)
        accessory_inventory_df = pd.read_csv(inventory_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure all CSV files are in the '{data_folder}/' directory.")
        return

    # 3. Prepare data
    print("Step 2: Preparing data...")
    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
    phones_df = products_df[products_df['product_type'] == 'Used Phone'].copy()
    
    # Filter for recent transactions to make the analysis faster
    cutoff_date = datetime.now() - timedelta(days=days_to_check)
    recent_phone_sales = transactions_df[
        (transactions_df['product_id'].isin(phones_df['product_id'])) &
        (transactions_df['transaction_date'] >= cutoff_date)
    ]

    # 4. Identify accessories to check (in stock with quantity > 0)
    accessories_in_stock = accessory_inventory_df[accessory_inventory_df['quantity'] > 0]
    
    discontinuation_list = []
    print(f"Step 3: Analyzing {len(accessories_in_stock['product_id'].unique())} unique in-stock accessories...")

    # 5. Iterate and analyze each accessory
    for acc_id in accessories_in_stock['product_id'].unique():
        # Find all phone models this accessory is compatible with
        compatible_phones = compatibility_df[compatibility_df['accessory_product_id'] == acc_id]
        
        if compatible_phones.empty:
            continue
            
        compatible_phone_ids = compatible_phones['phone_product_id'].tolist()
        
        # Calculate the total sales for those compatible phones in the last 180 days
        total_sales = recent_phone_sales[recent_phone_sales['product_id'].isin(compatible_phone_ids)].shape[0]
        
        # Check if the sales count is below the threshold
        if total_sales < sales_threshold:
            accessory_info = products_df[products_df['product_id'] == acc_id].iloc[0]
            discontinuation_list.append({
                'product_id': acc_id,
                'model_name': accessory_info['model_name'],
                'compatible_phone_sales_past_180_days': total_sales
            })

    # 6. Save the final list
    print(f"Step 4: Found {len(discontinuation_list)} accessories to recommend for discontinuation.")
    os.makedirs(output_folder, exist_ok=True)
    joblib.dump(discontinuation_list, output_path)
    
    print(f"\nDiscontinuation list built and saved successfully to '{output_path}'.")

if __name__ == "__main__":
    build_discontinuation_list()
