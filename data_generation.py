import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
from typing import Tuple
import os

# --- CONFIGURATION ---
START_DATE = datetime(2022, 8, 5)
END_DATE = datetime.now()
NUM_PHONE_UNITS = 14000
NUM_ACCESSORY_SALES = 11500 # Reduced for focus
SOLD_STATUS_PROBABILITY = 0.90
PHYSICAL_STORE_IDS = [1, 2, 3]
ALL_STORE_IDS = [1, 2, 3, 4]
OUTPUT_DIRECTORY = "data"

# Initialize Faker for data generation
fake = Faker()

# --- HELPER FUNCTIONS FOR PRICING & MARGIN LOGIC ---

def calculate_realistic_acquisition_price(
    product_details: pd.Series, acquisition_date: datetime.date, grade: str
) -> float:
    """Calculates a realistic acquisition price using a calibrated depreciation model."""
    base_price = product_details['original_msrp']
    used_market_value = base_price * 0.88
    days_old = max(0, (acquisition_date - product_details['release_date'].date()).days)
    daily_decay_rate = 0.00025
    used_market_value *= ((1 - daily_decay_rate) ** days_old)
    successor_release_date = product_details['successor_release_date']
    if pd.notna(successor_release_date) and acquisition_date > successor_release_date.date():
        used_market_value *= 0.80
    grade_acquisition_multipliers = {'A': 0.98, 'B': 0.90, 'C': 0.75, 'D': 0.60}
    acquisition_price_base = used_market_value * grade_acquisition_multipliers.get(grade, 0.5)
    final_price = acquisition_price_base * (1 + random.uniform(-0.03, 0.03))
    return round(max(0, final_price), 2)

def calculate_dynamic_max_margin(
    tier: float, grade: str, min_tier: float, max_tier: float
) -> float:
    """
    Calculates the maximum potential profit margin based on model age and grade.

    Args:
        tier: The model_tier of the phone.
        grade: The physical condition ('A', 'B', 'C', 'D').
        min_tier: The lowest tier in the entire product catalog.
        max_tier: The highest tier in the entire product catalog.

    Returns:
        The calculated maximum profit margin multiplier (e.g., 1.15 for 15% profit).
    """
    # 1. Define the profit margin range based on model newness.
    # Newest models can have up to 20% margin, oldest models as low as 8%.
    newest_tier_margin = 1.15
    oldest_tier_margin = 1.05

    # 2. Linearly interpolate to find the margin based on the phone's tier.
    # Normalize the current tier on a 0-1 scale
    if max_tier == min_tier: # Avoid division by zero if only one tier exists
        normalized_tier = 1.0
    else:
        normalized_tier = (tier - min_tier) / (max_tier - min_tier)
    
    tier_based_margin = oldest_tier_margin + normalized_tier * (newest_tier_margin - oldest_tier_margin)

    # 3. Adjust the calculated margin based on the phone's grade.
    # Grade A gets the full margin, while poorer grades get a reduced margin.
    grade_margin_adjusters = {'A': 1.0, 'B': 0.90, 'C': 0.75, 'D': 0.60}
    
    # We only adjust the "profit" part of the margin (the value above 1.0)
    profit_part = tier_based_margin - 1.0
    adjusted_profit = profit_part * grade_margin_adjusters.get(grade, 0.5)
    
    final_base_margin = 1.0 + adjusted_profit
    
    return final_base_margin


# --- DATA GENERATION FUNCTIONS ---

def generate_stores() -> pd.DataFrame:
    """Generates the stores table."""
    print("Generating stores table...")
    stores_data = [
        {'store_id': 1, 'store_name': 'Westfield Mall', 'location_type': 'Physical Store', 'city': 'Los Angeles'},
        {'store_id': 2, 'store_name': 'Beverly Center', 'location_type': 'Physical Store', 'city': 'Los Angeles'},
        {'store_id': 3, 'store_name': 'The Grove', 'location_type': 'Physical Store', 'city': 'Los Angeles'},
        {'store_id': 4, 'store_name': 'Online Store', 'location_type': 'Online', 'city': 'N/A'},
    ]
    return pd.DataFrame(stores_data)

def generate_products() -> pd.DataFrame:
    """Generates the master product catalog."""
    print("Generating products table...")
    phone_data = [

            # This list remains the same as v4

            # iPhone 11 series

            {'model_name': 'iPhone 11 64GB',        'model_tier': 11.0,  'storage_gb': 64,   'original_msrp': 699,  'release_date': '2019-09-20', 'successor_release_date': '2020-10-23'},

            {'model_name': 'iPhone 11 Pro 64GB',    'model_tier': 11.5,  'storage_gb': 64,   'original_msrp': 999,  'release_date': '2019-09-20', 'successor_release_date': '2020-10-23'},

            {'model_name': 'iPhone 11 Pro Max 64GB','model_tier': 11.7,  'storage_gb': 64,   'original_msrp': 1099, 'release_date': '2019-09-20', 'successor_release_date': '2020-10-23'},

            # iPhone 12 series

            {'model_name': 'iPhone 12 Mini 64GB',   'model_tier': 12.0, 'storage_gb': 64,   'original_msrp': 599,  'release_date': '2020-11-13', 'successor_release_date': '2021-09-24'},

            {'model_name': 'iPhone 12 64GB',        'model_tier': 12.2, 'storage_gb': 64,   'original_msrp': 799,  'release_date': '2020-10-23', 'successor_release_date': '2021-09-24'},

            {'model_name': 'iPhone 12 Pro 128GB',   'model_tier': 12.5, 'storage_gb': 128,  'original_msrp': 999,  'release_date': '2020-10-23', 'successor_release_date': '2021-09-24'},

            {'model_name': 'iPhone 12 Pro Max 128GB','model_tier': 12.7, 'storage_gb': 128,  'original_msrp': 1099, 'release_date': '2020-11-13', 'successor_release_date': '2021-09-24'},

            # iPhone 13 series

            {'model_name': 'iPhone 13 Mini 128GB',  'model_tier': 13.0, 'storage_gb': 128,  'original_msrp': 699,  'release_date': '2021-09-24', 'successor_release_date': '2022-09-16'},

            {'model_name': 'iPhone 13 128GB',       'model_tier': 13.2, 'storage_gb': 128,  'original_msrp': 799,  'release_date': '2021-09-24', 'successor_release_date': '2022-09-16'},

            {'model_name': 'iPhone 13 Pro 128GB',   'model_tier': 13.5, 'storage_gb': 128,  'original_msrp': 999,  'release_date': '2021-09-24', 'successor_release_date': '2022-09-16'},

            {'model_name': 'iPhone 13 Pro Max 128GB','model_tier': 13.7, 'storage_gb': 128,  'original_msrp': 1099, 'release_date': '2021-09-24', 'successor_release_date': '2022-09-16'},

            # iPhone 14 series

            {'model_name': 'iPhone 14 128GB',       'model_tier': 14.0, 'storage_gb': 128,  'original_msrp': 799,  'release_date': '2022-09-16', 'successor_release_date': '2023-09-22'},

            {'model_name': 'iPhone 14 Plus 128GB',  'model_tier': 14.2, 'storage_gb': 128,  'original_msrp': 899,  'release_date': '2022-10-07', 'successor_release_date': '2023-09-22'},

            {'model_name': 'iPhone 14 Pro 128GB',   'model_tier': 14.5, 'storage_gb': 128,  'original_msrp': 1099, 'release_date': '2022-09-16', 'successor_release_date': '2023-09-22'},

            {'model_name': 'iPhone 14 Pro Max 128GB','model_tier': 14.7, 'storage_gb': 128,  'original_msrp': 1199, 'release_date': '2022-09-16', 'successor_release_date': '2023-09-22'},

            # iPhone 15 series

            {'model_name': 'iPhone 15 256GB',       'model_tier': 15.0, 'storage_gb': 256,  'original_msrp': 899,  'release_date': '2023-09-22', 'successor_release_date': '2024-09-20'},

            {'model_name': 'iPhone 15 Plus 256GB',  'model_tier': 15.2, 'storage_gb': 256,  'original_msrp': 999,  'release_date': '2023-09-22', 'successor_release_date': '2024-09-20'},

            {'model_name': 'iPhone 15 Pro 256GB',   'model_tier': 15.5, 'storage_gb': 256,  'original_msrp': 1199, 'release_date': '2023-09-22', 'successor_release_date': '2024-09-20'},

            {'model_name': 'iPhone 15 Pro Max 256GB','model_tier': 15.7, 'storage_gb': 256,  'original_msrp': 1399, 'release_date': '2023-09-22', 'successor_release_date': '2024-09-20'},

            # iPhone 16 series

            {'model_name': 'iPhone 16 128GB',       'model_tier': 16.0, 'storage_gb': 128,  'original_msrp': 829,  'release_date': '2024-09-20', 'successor_release_date': None},

            {'model_name': 'iPhone 16 Plus 128GB',  'model_tier': 16.2, 'storage_gb': 128,  'original_msrp': 929,  'release_date': '2024-09-20', 'successor_release_date': None},

            {'model_name': 'iPhone 16 Pro 128GB',   'model_tier': 16.5, 'storage_gb': 128,  'original_msrp': 999,  'release_date': '2024-09-20', 'successor_release_date': None},

            {'model_name': 'iPhone 16 Pro Max 256GB','model_tier': 16.7, 'storage_gb': 256,  'original_msrp': 1199, 'release_date': '2024-09-20', 'successor_release_date': None},

        ]
    for p in phone_data: p['product_type'] = 'Used Phone'
    accessory_data = [{'model_name': '20W USB-C Power Adapter', 'original_msrp': 19, 'compatible_tier': 'all'}]
    for p in accessory_data: p['product_type'] = 'Accessory'
    products_list = phone_data + accessory_data
    df = pd.DataFrame(products_list)
    df['product_id'] = range(1, len(df) + 1)
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['successor_release_date'] = pd.to_datetime(df['successor_release_date'])
    return df

def generate_inventory_units(products_df: pd.DataFrame) -> pd.DataFrame:
    """Generates inventory units, including `age_at_acquisition_days`."""
    print(f"Generating {NUM_PHONE_UNITS} unique inventory units...")
    inventory_list = []
    phone_products = products_df[products_df['product_type'] == 'Used Phone'].copy()
    phone_products.set_index('product_id', inplace=True)
    total_days = (END_DATE - START_DATE).days
    random_days = np.random.randint(0, total_days, size=NUM_PHONE_UNITS)
    acquisition_dates = [START_DATE + timedelta(days=int(d)) for d in random_days]
    for i in range(NUM_PHONE_UNITS):
        unit_id = i + 1
        product_id = random.choice(phone_products.index)
        product_details = phone_products.loc[product_id]
        acquisition_date = random.choice(acquisition_dates).date()
        while acquisition_date < product_details['release_date'].date():
            acquisition_date = (START_DATE + timedelta(days=random.randint(0, total_days))).date()
        grade = random.choices(['A', 'B', 'C', 'D'], weights=[0.4, 0.35, 0.2, 0.05], k=1)[0]
        age_at_acquisition_days = (acquisition_date - product_details['release_date'].date()).days
        acquisition_price = calculate_realistic_acquisition_price(product_details, acquisition_date, grade)
        inventory_list.append({
            'unit_id': unit_id, 'product_id': product_id,
            'store_id': random.choice(PHYSICAL_STORE_IDS),
            'acquisition_date': acquisition_date, 'grade': grade,
            'age_at_acquisition_days': age_at_acquisition_days,
            'acquisition_price': acquisition_price,
            'status': 'Sold' if random.random() < SOLD_STATUS_PROBABILITY else 'In Stock',
        })
    return pd.DataFrame(inventory_list)

def generate_transactions(
    inventory_units_df: pd.DataFrame, products_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generates sales transactions with multi-factor dynamic profit margins.
    """
    print("Generating transactions table (with multi-factor dynamic margins)...")
    transactions_list = []
    transaction_id_counter = 1

    # Get min/max tiers for margin calculation
    phone_products = products_df[products_df['product_type'] == 'Used Phone']
    min_tier = phone_products['model_tier'].min()
    max_tier = phone_products['model_tier'].max()
    
    sold_phones = inventory_units_df[inventory_units_df['status'] == 'Sold']
    
    # Merge to get all necessary details in one place
    sold_phones_with_details = pd.merge(
        sold_phones,
        products_df[['product_id', 'release_date', 'original_msrp', 'model_tier']],
        on='product_id'
    )

    for _, unit in sold_phones_with_details.iterrows():
        days_on_market = random.randint(1, 365)
        transaction_date = unit['acquisition_date'] + timedelta(days=days_on_market)
        if transaction_date > END_DATE.date():
            transaction_date = END_DATE.date()
        
        age_at_sale_days = (transaction_date - unit['release_date'].date()).days
        
        # --- NEW: Multi-Factor Margin Calculation ---
        # 1. Calculate the base maximum margin for this specific phone.
        max_profit_margin = calculate_dynamic_max_margin(
            tier=unit['model_tier'],
            grade=unit['grade'],
            min_tier=min_tier,
            max_tier=max_tier
        )

        # 2. Apply the decay based on days_on_market to the dynamic margin.
        margin_decay_per_day = 0.0005
        floor_margin = 0.90
        dynamic_profit_margin = max_profit_margin - (days_on_market * margin_decay_per_day)
        final_profit_margin = max(floor_margin, dynamic_profit_margin)
        
        # 3. Calculate final price.
        calculated_sale_price = unit['acquisition_price'] * final_profit_margin
        final_sale_price = min(calculated_sale_price, unit['original_msrp'])

        transactions_list.append({
            'transaction_id': transaction_id_counter, 'unit_id': unit['unit_id'],
            'product_id': unit['product_id'], 'store_id': random.choice(ALL_STORE_IDS),
            'transaction_date': transaction_date,
            'age_at_sale_days': age_at_sale_days,
            'final_sale_price': round(final_sale_price, 2),
            'transaction_type': 'Sale'
        })
        transaction_id_counter += 1

    return pd.DataFrame(transactions_list)

# --- MAIN ORCHESTRATION FUNCTION ---

def main():
    """Main function to orchestrate the data generation and save files."""
    print("--- Starting Synthetic Data Generation (v9 - Multi-Factor Dynamic Margins) ---")
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
        print(f"Created output directory: {OUTPUT_DIRECTORY}/")
    
    stores_df = generate_stores()
    products_df = generate_products()
    inventory_units_df = generate_inventory_units(products_df)
    transactions_df = generate_transactions(inventory_units_df, products_df)
    
    print(f"\nSaving dataframes to '{OUTPUT_DIRECTORY}/' directory...")
    file_paths = {
        "stores.csv": stores_df, "products.csv": products_df,
        "inventory_units.csv": inventory_units_df, "transactions.csv": transactions_df
    }
    for filename, df in file_paths.items():
        try:
            full_path = os.path.join(OUTPUT_DIRECTORY, filename)
            df.to_csv(full_path, index=False)
            print(f"- Successfully saved {full_path}")
        except Exception as e:
            print(f"An error occurred while saving {filename}: {e}")
    print("\n--- Data Generation Complete ---")

if __name__ == "__main__":
    main()