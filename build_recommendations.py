import pandas as pd
import os
import joblib
from itertools import combinations

def build_recommendation_data():
    """
    Pre-calculates all data needed for the hybrid recommendation engine,
    including overall popularity and co-purchase counts (market basket analysis).
    """
    print("--- Building Advanced Recommendation Data ---")
    
    # 1. Define file paths
    data_folder = 'data'
    products_path = os.path.join(data_folder, 'products.csv')
    transactions_path = os.path.join(data_folder, 'transactions.csv')
    compatibility_path = os.path.join(data_folder, 'accessory_compatibility.csv')
    output_folder = 'models'
    output_path = os.path.join(output_folder, 'recommendation_data_advanced.joblib')

    # 2. Load all necessary data
    print("Step 1: Loading data...")
    try:
        products_df = pd.read_csv(products_path)
        transactions_df = pd.read_csv(transactions_path)
        compatibility_df = pd.read_csv(compatibility_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure all CSV files are in the '{data_folder}/' directory.")
        return
    
    # --- Part 1: Calculate Overall Popularity (Same as before) ---
    print("Step 2: Calculating overall accessory popularity...")
    accessories = products_df[products_df['product_type'] == 'Accessory']
    accessory_sales = transactions_df[transactions_df['product_id'].isin(accessories['product_id'])]
    accessory_popularity = accessory_sales['product_id'].value_counts()

    # --- Part 2: Calculate Co-purchase Counts (Market Basket Analysis) ---
    print("Step 3: Performing market basket analysis for co-purchases...")
    # Add product type and base model info to transactions
    phones = products_df[products_df['product_type'] == 'Used Phone'].copy()
    phones['base_model'] = phones['model_name'].str.extract(r'(iPhone \d{1,2}(?: Pro| Plus| Pro Max| Mini)?)')[0]
    
    transactions_with_types = pd.merge(
        transactions_df, 
        products_df[['product_id', 'product_type']], 
        on='product_id'
    )
    # Add phone base_model to phone transactions for easier grouping
    transactions_with_models = pd.merge(
        transactions_with_types,
        phones[['product_id', 'base_model']],
        on='product_id',
        how='left'
    )
    
    # Group by transaction_id to find items bought together
    baskets = transactions_with_models.groupby('transaction_id')
    
    co_purchase_counts = {} # Will store: {('phone_base_model', 'accessory_id'): count}

    for _, basket in baskets:
        # Separate phones and accessories in the basket
        phone_in_basket = basket[basket['product_type'] == 'Used Phone']
        accessories_in_basket = basket[basket['product_type'] == 'Accessory']
        
        # If there's at least one phone and one accessory, log the co-purchase
        if not phone_in_basket.empty and not accessories_in_basket.empty:
            phone_base_model = phone_in_basket['base_model'].iloc[0] # Assume one phone per transaction for simplicity
            for acc_id in accessories_in_basket['product_id']:
                key = (phone_base_model, acc_id)
                co_purchase_counts[key] = co_purchase_counts.get(key, 0) + 1

    # Convert the dictionary to a more usable DataFrame
    co_purchase_df = pd.DataFrame(
        list(co_purchase_counts.items()), 
        columns=['key', 'count']
    )
    co_purchase_df[['phone_base_model', 'accessory_product_id']] = pd.DataFrame(co_purchase_df['key'].tolist(), index=co_purchase_df.index)
    co_purchase_df.drop(columns=['key'], inplace=True)


    # 4. Package all data for saving
    print("Step 4: Packaging all data...")
    recommendation_package = {
        'popularity': accessory_popularity,
        'compatibility': compatibility_df,
        'co_purchase': co_purchase_df
    }
    
    # 5. Save the packaged data
    print(f"Step 5: Saving advanced recommendation data to '{output_path}'...")
    os.makedirs(output_folder, exist_ok=True)
    joblib.dump(recommendation_package, output_path)
    
    print("\nAdvanced recommendation data built and saved successfully.")

if __name__ == "__main__":
    build_recommendation_data()
