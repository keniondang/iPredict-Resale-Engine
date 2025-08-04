# -*- coding: utf-8 -*-
"""
Synthetic Data Generation for a Used iPhone Retail Business (v4 - Realistic Acquisition Pricing)

This script generates a high-fidelity, interconnected dataset for a business
that buys and sells used iPhones and accessories.

VERSION 4 UPDATES:
- Overhauled the `calculate_realistic_acquisition_price` function for greater realism.
- Introduced a two-stage model: 1) Calculate used market value (with immediate depreciation),
  and 2) Calculate business acquisition price based on that value to ensure profit margin.
- This prevents acquisition prices from being too close to MSRP and creating market anomalies.
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
from typing import Tuple
import os

# --- CONFIGURATION ---
START_DATE = datetime(2022, 8, 5)
END_DATE = datetime(2025, 8, 5) # Updated to current date
NUM_PHONE_UNITS = 4000
NUM_ACCESSORY_SALES = 7500
SOLD_STATUS_PROBABILITY = 0.90
PHYSICAL_STORE_IDS = [1, 2, 3]
ALL_STORE_IDS = [1, 2, 3, 4]
OUTPUT_DIRECTORY = "data"

# Initialize Faker for data generation
fake = Faker()

# --- HELPER FUNCTION FOR PRICING LOGIC (v4) ---

def calculate_realistic_acquisition_price(
    product_details: pd.Series, acquisition_date: datetime.date, grade: str
) -> float:
    """
    Calculates a realistic acquisition price using a sophisticated two-stage model.
    This ensures acquisition prices are realistic and allow for profit margins.

    Args:
        product_details: A pandas Series with product info.
        acquisition_date: The date the business acquired the phone.
        grade: The condition of the phone ('A', 'B', 'C', 'D').

    Returns:
        The calculated acquisition price.
    """
    base_price = product_details['original_msrp']

    # --- Stage 1: Establish the "Used Market Value" ---
    # This is what a customer might expect to pay for a used item.

    # 1a. IMMEDIATE DEPRECIATION: A phone loses significant value the moment it's unboxed.
    used_market_value = base_price * 0.85  # Immediate 15% drop for being "used"

    # 1b. AGE DEPRECIATION: Further decay over time from the new baseline.
    days_old = max(0, (acquisition_date - product_details['release_date'].date()).days)
    daily_decay_rate = 0.0006  # A slightly adjusted daily decay
    used_market_value *= ((1 - daily_decay_rate) ** days_old)

    # 1c. SUCCESSOR DEPRECIATION: Sharp drop when a new model is released.
    successor_release_date = product_details['successor_release_date']
    if pd.notna(successor_release_date) and acquisition_date > successor_release_date.date():
        used_market_value *= 0.80  # Additional 20% drop

    # --- Stage 2: Calculate the Business's Acquisition Price ---
    # This is based on the used market value, grade, and the need for a profit margin.

    # 2a. GRADE-BASED OFFER: The business will not offer 100% of the used market value.
    # This multiplier builds in the profit margin at the point of acquisition.
    grade_acquisition_multipliers = {'A': 0.95, 'B': 0.85, 'C': 0.70, 'D': 0.55}
    acquisition_price_base = used_market_value * grade_acquisition_multipliers.get(grade, 0.5)

    # 2b. Add noise for negotiation or minor market fluctuations.
    final_price = acquisition_price_base * (1 + random.uniform(-0.04, 0.04))

    # Ensure price is never negative and round it.
    return round(max(0, final_price), 2)


# --- DATA GENERATION FUNCTIONS ---

def generate_stores() -> pd.DataFrame:
    """Generates the stores table with physical and online locations."""
    print("Generating stores table...")
    stores_data = [
        {'store_id': 1, 'store_name': 'Westfield Mall', 'location_type': 'Physical Store', 'city': 'Los Angeles'},
        {'store_id': 2, 'store_name': 'Beverly Center', 'location_type': 'Physical Store', 'city': 'Los Angeles'},
        {'store_id': 3, 'store_name': 'The Grove', 'location_type': 'Physical Store', 'city': 'Los Angeles'},
        {'store_id': 4, 'store_name': 'Online Store', 'location_type': 'Online', 'city': 'N/A'},
    ]
    return pd.DataFrame(stores_data)

def generate_products() -> pd.DataFrame:
    """Generates the master product catalog for iPhones and accessories."""
    print("Generating products table with expanded list...")
    phone_data = [
        # iPhone 11 series
        {'model_name': 'iPhone 11 64GB',        'model_tier': 11.0,  'storage_gb': 64,   'original_msrp': 699,  'release_date': '2019-09-20', 'successor_release_date': '2020-10-23'},
        {'model_name': 'iPhone 11 128GB',       'model_tier': 11.0,  'storage_gb': 128,  'original_msrp': 749,  'release_date': '2019-09-20', 'successor_release_date': '2020-10-23'},
        {'model_name': 'iPhone 11 256GB',       'model_tier': 11.0,  'storage_gb': 256,  'original_msrp': 849,  'release_date': '2019-09-20', 'successor_release_date': '2020-10-23'},
        {'model_name': 'iPhone 11 Pro 64GB',    'model_tier': 11.5,  'storage_gb': 64,   'original_msrp': 999,  'release_date': '2019-09-20', 'successor_release_date': '2020-10-23'},
        {'model_name': 'iPhone 11 Pro 256GB',   'model_tier': 11.5,  'storage_gb': 256,  'original_msrp': 1149, 'release_date': '2019-09-20', 'successor_release_date': '2020-10-23'},
        {'model_name': 'iPhone 11 Pro 512GB',   'model_tier': 11.5,  'storage_gb': 512,  'original_msrp': 1349, 'release_date': '2019-09-20', 'successor_release_date': '2020-10-23'},
        {'model_name': 'iPhone 11 Pro Max 64GB','model_tier': 11.7,  'storage_gb': 64,   'original_msrp': 1099, 'release_date': '2019-09-20', 'successor_release_date': '2020-10-23'},
        {'model_name': 'iPhone 11 Pro Max 256GB','model_tier': 11.7, 'storage_gb': 256,  'original_msrp': 1249, 'release_date': '2019-09-20', 'successor_release_date': '2020-10-23'},
        {'model_name': 'iPhone 11 Pro Max 512GB','model_tier': 11.7, 'storage_gb': 512,  'original_msrp': 1449, 'release_date': '2019-09-20', 'successor_release_date': '2020-10-23'},
        # iPhone 12 series
        {'model_name': 'iPhone 12 Mini 64GB',   'model_tier': 12.0, 'storage_gb': 64,   'original_msrp': 599,  'release_date': '2020-11-13', 'successor_release_date': '2021-09-24'},
        {'model_name': 'iPhone 12 Mini 128GB',  'model_tier': 12.0, 'storage_gb': 128,  'original_msrp': 749,  'release_date': '2020-11-13', 'successor_release_date': '2021-09-24'},
        {'model_name': 'iPhone 12 Mini 256GB',  'model_tier': 12.0, 'storage_gb': 256,  'original_msrp': 849,  'release_date': '2020-11-13', 'successor_release_date': '2021-09-24'},
        {'model_name': 'iPhone 12 64GB',        'model_tier': 12.2, 'storage_gb': 64,   'original_msrp': 799,  'release_date': '2020-10-23', 'successor_release_date': '2021-09-24'},
        {'model_name': 'iPhone 12 128GB',       'model_tier': 12.2, 'storage_gb': 128,  'original_msrp': 849,  'release_date': '2020-10-23', 'successor_release_date': '2021-09-24'},
        {'model_name': 'iPhone 12 256GB',       'model_tier': 12.2, 'storage_gb': 256,  'original_msrp': 949,  'release_date': '2020-10-23', 'successor_release_date': '2021-09-24'},
        {'model_name': 'iPhone 12 Pro 128GB',   'model_tier': 12.5, 'storage_gb': 128,  'original_msrp': 999,  'release_date': '2020-10-23', 'successor_release_date': '2021-09-24'},
        {'model_name': 'iPhone 12 Pro 256GB',   'model_tier': 12.5, 'storage_gb': 256,  'original_msrp': 1099, 'release_date': '2020-10-23', 'successor_release_date': '2021-09-24'},
        {'model_name': 'iPhone 12 Pro 512GB',   'model_tier': 12.5, 'storage_gb': 512,  'original_msrp': 1299, 'release_date': '2020-10-23', 'successor_release_date': '2021-09-24'},
        {'model_name': 'iPhone 12 Pro Max 128GB','model_tier': 12.7, 'storage_gb': 128,  'original_msrp': 1099, 'release_date': '2020-11-13', 'successor_release_date': '2021-09-24'},
        {'model_name': 'iPhone 12 Pro Max 256GB','model_tier': 12.7, 'storage_gb': 256,  'original_msrp': 1199, 'release_date': '2020-11-13', 'successor_release_date': '2021-09-24'},
        {'model_name': 'iPhone 12 Pro Max 512GB','model_tier': 12.7, 'storage_gb': 512,  'original_msrp': 1399, 'release_date': '2020-11-13', 'successor_release_date': '2021-09-24'},
        # iPhone 13 series
        {'model_name': 'iPhone 13 Mini 128GB',  'model_tier': 13.0, 'storage_gb': 128,  'original_msrp': 699,  'release_date': '2021-09-24', 'successor_release_date': '2022-09-16'},
        {'model_name': 'iPhone 13 Mini 256GB',  'model_tier': 13.0, 'storage_gb': 256,  'original_msrp': 799,  'release_date': '2021-09-24', 'successor_release_date': '2022-09-16'},
        {'model_name': 'iPhone 13 Mini 512GB',  'model_tier': 13.0, 'storage_gb': 512,  'original_msrp': 999,  'release_date': '2021-09-24', 'successor_release_date': '2022-09-16'},
        {'model_name': 'iPhone 13 128GB',       'model_tier': 13.2, 'storage_gb': 128,  'original_msrp': 799,  'release_date': '2021-09-24', 'successor_release_date': '2022-09-16'},
        {'model_name': 'iPhone 13 256GB',       'model_tier': 13.2, 'storage_gb': 256,  'original_msrp': 899,  'release_date': '2021-09-24', 'successor_release_date': '2022-09-16'},
        {'model_name': 'iPhone 13 512GB',       'model_tier': 13.2, 'storage_gb': 512,  'original_msrp': 1099, 'release_date': '2021-09-24', 'successor_release_date': '2022-09-16'},
        {'model_name': 'iPhone 13 Pro 128GB',   'model_tier': 13.5, 'storage_gb': 128,  'original_msrp': 999,  'release_date': '2021-09-24', 'successor_release_date': '2022-09-16'},
        {'model_name': 'iPhone 13 Pro 256GB',   'model_tier': 13.5, 'storage_gb': 256,  'original_msrp': 1099, 'release_date': '2021-09-24', 'successor_release_date': '2022-09-16'},
        {'model_name': 'iPhone 13 Pro 512GB',   'model_tier': 13.5, 'storage_gb': 512,  'original_msrp': 1299, 'release_date': '2021-09-24', 'successor_release_date': '2022-09-16'},
        {'model_name': 'iPhone 13 Pro 1TB',     'model_tier': 13.5, 'storage_gb': 1024, 'original_msrp': 1499, 'release_date': '2021-09-24', 'successor_release_date': '2022-09-16'},
        {'model_name': 'iPhone 13 Pro Max 128GB','model_tier': 13.7, 'storage_gb': 128,  'original_msrp': 1099, 'release_date': '2021-09-24', 'successor_release_date': '2022-09-16'},
        {'model_name': 'iPhone 13 Pro Max 256GB','model_tier': 13.7, 'storage_gb': 256,  'original_msrp': 1199, 'release_date': '2021-09-24', 'successor_release_date': '2022-09-16'},
        {'model_name': 'iPhone 13 Pro Max 512GB','model_tier': 13.7, 'storage_gb': 512,  'original_msrp': 1399, 'release_date': '2021-09-24', 'successor_release_date': '2022-09-16'},
        {'model_name': 'iPhone 13 Pro Max 1TB', 'model_tier': 13.7, 'storage_gb': 1024, 'original_msrp': 1499, 'release_date': '2021-09-24', 'successor_release_date': '2022-09-16'},
        # iPhone 14 series
        {'model_name': 'iPhone 14 128GB',       'model_tier': 14.0, 'storage_gb': 128,  'original_msrp': 799,  'release_date': '2022-09-16', 'successor_release_date': '2023-09-22'},
        {'model_name': 'iPhone 14 256GB',       'model_tier': 14.0, 'storage_gb': 256,  'original_msrp': 899,  'release_date': '2022-09-16', 'successor_release_date': '2023-09-22'},
        {'model_name': 'iPhone 14 512GB',       'model_tier': 14.0, 'storage_gb': 512,  'original_msrp': 1099, 'release_date': '2022-09-16', 'successor_release_date': '2023-09-22'},
        {'model_name': 'iPhone 14 Plus 128GB',  'model_tier': 14.2, 'storage_gb': 128,  'original_msrp': 899,  'release_date': '2022-10-07', 'successor_release_date': '2023-09-22'},
        {'model_name': 'iPhone 14 Plus 256GB',  'model_tier': 14.2, 'storage_gb': 256,  'original_msrp': 999,  'release_date': '2022-10-07', 'successor_release_date': '2023-09-22'},
        {'model_name': 'iPhone 14 Plus 512GB',  'model_tier': 14.2, 'storage_gb': 512,  'original_msrp': 1199, 'release_date': '2022-10-07', 'successor_release_date': '2023-09-22'},
        {'model_name': 'iPhone 14 Pro 128GB',   'model_tier': 14.5, 'storage_gb': 128,  'original_msrp': 1099, 'release_date': '2022-09-16', 'successor_release_date': '2023-09-22'},
        {'model_name': 'iPhone 14 Pro 256GB',   'model_tier': 14.5, 'storage_gb': 256,  'original_msrp': 1199, 'release_date': '2022-09-16', 'successor_release_date': '2023-09-22'},
        {'model_name': 'iPhone 14 Pro 512GB',   'model_tier': 14.5, 'storage_gb': 512,  'original_msrp': 1399, 'release_date': '2022-09-16', 'successor_release_date': '2023-09-22'},
        {'model_name': 'iPhone 14 Pro 1TB',     'model_tier': 14.5, 'storage_gb': 1024, 'original_msrp': 1599, 'release_date': '2022-09-16', 'successor_release_date': '2023-09-22'},
        {'model_name': 'iPhone 14 Pro Max 128GB','model_tier': 14.7, 'storage_gb': 128,  'original_msrp': 1199, 'release_date': '2022-09-16', 'successor_release_date': '2023-09-22'},
        {'model_name': 'iPhone 14 Pro Max 256GB','model_tier': 14.7, 'storage_gb': 256,  'original_msrp': 1299, 'release_date': '2022-09-16', 'successor_release_date': '2023-09-22'},
        {'model_name': 'iPhone 14 Pro Max 512GB','model_tier': 14.7, 'storage_gb': 512,  'original_msrp': 1499, 'release_date': '2022-09-16', 'successor_release_date': '2023-09-22'},
        {'model_name': 'iPhone 14 Pro Max 1TB', 'model_tier': 14.7, 'storage_gb': 1024, 'original_msrp': 1699, 'release_date': '2022-09-16', 'successor_release_date': '2023-09-22'},
        # iPhone 15 series
        {'model_name': 'iPhone 15 256GB',       'model_tier': 15.0, 'storage_gb': 256,  'original_msrp': 899,  'release_date': '2023-09-22', 'successor_release_date': '2024-09-20'},
        {'model_name': 'iPhone 15 512GB',       'model_tier': 15.0, 'storage_gb': 512,  'original_msrp': 1099, 'release_date': '2023-09-22', 'successor_release_date': '2024-09-20'},
        {'model_name': 'iPhone 15 Plus 256GB',  'model_tier': 15.2, 'storage_gb': 256,  'original_msrp': 999,  'release_date': '2023-09-22', 'successor_release_date': '2024-09-20'},
        {'model_name': 'iPhone 15 Pro 256GB',   'model_tier': 15.5, 'storage_gb': 256,  'original_msrp': 1199, 'release_date': '2023-09-22', 'successor_release_date': '2024-09-20'},
        {'model_name': 'iPhone 15 Pro 512GB',   'model_tier': 15.5, 'storage_gb': 512,  'original_msrp': 1399, 'release_date': '2023-09-22', 'successor_release_date': '2024-09-20'},
        {'model_name': 'iPhone 15 Pro 1TB',     'model_tier': 15.5, 'storage_gb': 1024, 'original_msrp': 1599, 'release_date': '2023-09-22', 'successor_release_date': '2024-09-20'},
        {'model_name': 'iPhone 15 Pro Max 256GB','model_tier': 15.7, 'storage_gb': 256,  'original_msrp': 1399, 'release_date': '2023-09-22', 'successor_release_date': '2024-09-20'},
        {'model_name': 'iPhone 15 Pro Max 512GB','model_tier': 15.7, 'storage_gb': 512,  'original_msrp': 1599, 'release_date': '2023-09-22', 'successor_release_date': '2024-09-20'},
        {'model_name': 'iPhone 15 Pro Max 1TB', 'model_tier': 15.7, 'storage_gb': 1024, 'original_msrp': 1799, 'release_date': '2023-09-22', 'successor_release_date': '2024-09-20'},
        # iPhone 16 series
        {'model_name': 'iPhone 16 128GB',       'model_tier': 16.0, 'storage_gb': 128,  'original_msrp': 829,  'release_date': '2024-09-20', 'successor_release_date': None},
        {'model_name': 'iPhone 16 256GB',       'model_tier': 16.0, 'storage_gb': 256,  'original_msrp': 899,  'release_date': '2024-09-20', 'successor_release_date': None},
        {'model_name': 'iPhone 16 512GB',       'model_tier': 16.0, 'storage_gb': 512,  'original_msrp': 999,  'release_date': '2024-09-20', 'successor_release_date': None},
        {'model_name': 'iPhone 16 Plus 128GB',  'model_tier': 16.2, 'storage_gb': 128,  'original_msrp': 929,  'release_date': '2024-09-20', 'successor_release_date': None},
        {'model_name': 'iPhone 16 Plus 256GB',  'model_tier': 16.2, 'storage_gb': 256,  'original_msrp': 999,  'release_date': '2024-09-20', 'successor_release_date': None},
        {'model_name': 'iPhone 16 Plus 512GB',  'model_tier': 16.2, 'storage_gb': 512,  'original_msrp': 1099, 'release_date': '2024-09-20', 'successor_release_date': None},
        {'model_name': 'iPhone 16 Pro 128GB',   'model_tier': 16.5, 'storage_gb': 128,  'original_msrp': 999,  'release_date': '2024-09-20', 'successor_release_date': None},
        {'model_name': 'iPhone 16 Pro 256GB',   'model_tier': 16.5, 'storage_gb': 256,  'original_msrp': 1099, 'release_date': '2024-09-20', 'successor_release_date': None},
        {'model_name': 'iPhone 16 Pro 512GB',   'model_tier': 16.5, 'storage_gb': 512,  'original_msrp': 1299, 'release_date': '2024-09-20', 'successor_release_date': None},
        {'model_name': 'iPhone 16 Pro 1TB',     'model_tier': 16.5, 'storage_gb': 1024, 'original_msrp': 1499, 'release_date': '2024-09-20', 'successor_release_date': None},
        {'model_name': 'iPhone 16 Pro Max 256GB','model_tier': 16.7, 'storage_gb': 256,  'original_msrp': 1199, 'release_date': '2024-09-20', 'successor_release_date': None},
        {'model_name': 'iPhone 16 Pro Max 512GB','model_tier': 16.7, 'storage_gb': 512,  'original_msrp': 1399, 'release_date': '2024-09-20', 'successor_release_date': None},
        {'model_name': 'iPhone 16 Pro Max 1TB', 'model_tier': 16.7, 'storage_gb': 1024, 'original_msrp': 1599, 'release_date': '2024-09-20', 'successor_release_date': None},
    ]
    for p in phone_data:
        p['product_type'] = 'Used Phone'

    accessory_data = [
        {'model_name': '20W USB-C Power Adapter', 'original_msrp': 19, 'compatible_tier': 'all'},
        {'model_name': 'MagSafe Charger', 'original_msrp': 39, 'compatible_tier': 'all_magsafe'},
        {'model_name': 'Lightning to USB-C Cable (1m)', 'original_msrp': 19, 'compatible_tier': 'all_lightning'},
        {'model_name': 'USB-C to USB-C Cable (1m)', 'original_msrp': 19, 'compatible_tier': 'all_usbc'},
        {'model_name': 'AirPods Pro (2nd generation)', 'original_msrp': 249, 'compatible_tier': 'all'},
    ]
    for p in accessory_data:
        p['product_type'] = 'Accessory'

    products_list = phone_data + accessory_data
    df = pd.DataFrame(products_list)
    df['product_id'] = range(1, len(df) + 1)
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['successor_release_date'] = pd.to_datetime(df['successor_release_date'])
    return df

def generate_inventory_units(products_df: pd.DataFrame) -> pd.DataFrame:
    """Generates unique inventory units for used phones."""
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
        
        acquisition_date = random.choice(acquisition_dates)
        while acquisition_date.date() < product_details['release_date'].date():
            acquisition_date = START_DATE + timedelta(days=random.randint(0, total_days))
        
        grade = random.choices(['A', 'B', 'C', 'D'], weights=[0.4, 0.35, 0.2, 0.05], k=1)[0]
        acquisition_price = calculate_realistic_acquisition_price(product_details, acquisition_date.date(), grade)

        inventory_list.append({
            'unit_id': unit_id,
            'product_id': product_id,
            'store_id': random.choice(PHYSICAL_STORE_IDS),
            'acquisition_date': acquisition_date.date(),
            'grade': grade,
            'acquisition_price': acquisition_price,
            'status': 'Sold' if random.random() < SOLD_STATUS_PROBABILITY else 'In Stock',
        })

    return pd.DataFrame(inventory_list)

def generate_transactions(
    inventory_units_df: pd.DataFrame, products_df: pd.DataFrame
) -> pd.DataFrame:
    """Generates sales transactions, capping phone prices at original MSRP."""
    print("Generating transactions table (with price capping)...")
    transactions_list = []
    transaction_id_counter = 1

    products_indexed = products_df.set_index('product_id')
    sold_phones = inventory_units_df[inventory_units_df['status'] == 'Sold']
    
    for _, unit in sold_phones.iterrows():
        days_to_sell = random.randint(1, 180)
        transaction_date = unit['acquisition_date'] + timedelta(days=days_to_sell)
        
        if transaction_date > END_DATE.date():
            transaction_date = END_DATE.date() - timedelta(days=random.randint(1,5))

        original_msrp = products_indexed.loc[unit['product_id'], 'original_msrp']
        potential_sale_price = unit['acquisition_price'] * random.uniform(1.15, 1.40)
        final_price = min(potential_sale_price, original_msrp)

        transactions_list.append({
            'transaction_id': transaction_id_counter,
            'unit_id': unit['unit_id'],
            'product_id': unit['product_id'],
            'store_id': random.choice(ALL_STORE_IDS),
            'transaction_date': transaction_date,
            'final_sale_price': round(final_price, 2),
            'transaction_type': 'Sale'
        })
        transaction_id_counter += 1

    accessory_products = products_df[products_df['product_type'] == 'Accessory']
    total_days = (END_DATE - START_DATE).days
    accessory_product_ids = accessory_products['product_id'].tolist()
    
    for _ in range(NUM_ACCESSORY_SALES):
        product_id = random.choice(accessory_product_ids)
        product_details = products_indexed.loc[product_id]
        
        transactions_list.append({
            'transaction_id': transaction_id_counter,
            'unit_id': np.nan,
            'product_id': product_id,
            'store_id': random.choice(ALL_STORE_IDS),
            'transaction_date': START_DATE.date() + timedelta(days=random.randint(0, total_days)),
            'final_sale_price': round(product_details['original_msrp'] * random.uniform(0.98, 1.05), 2),
            'transaction_type': 'Sale'
        })
        transaction_id_counter += 1

    return pd.DataFrame(transactions_list)

def generate_accessory_data(
    products_df: pd.DataFrame, stores_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates accessory inventory levels and a compatibility map."""
    print("Generating accessory inventory and compatibility tables...")
    
    accessory_inventory_list = []
    accessory_products = products_df[products_df['product_type'] == 'Accessory']
    physical_stores = stores_df[stores_df['location_type'] == 'Physical Store']
    
    for _, store in physical_stores.iterrows():
        for _, accessory in accessory_products.iterrows():
            accessory_inventory_list.append({
                'store_id': store['store_id'],
                'product_id': accessory['product_id'],
                'quantity': random.randint(10, 100)
            })
    accessory_inventory_df = pd.DataFrame(accessory_inventory_list)
    
    compatibility_list = []
    phone_products = products_df[products_df['product_type'] == 'Used Phone']
    
    for _, phone in phone_products.iterrows():
        if phone['model_tier'] >= 12:
            magsafe_accessories = products_df[products_df['compatible_tier'] == 'all_magsafe']
            for _, acc in magsafe_accessories.iterrows():
                compatibility_list.append({'phone_product_id': phone['product_id'], 'accessory_product_id': acc['product_id']})
        if phone['model_tier'] >= 15:
            usbc_accessories = products_df[products_df['compatible_tier'] == 'all_usbc']
            for _, acc in usbc_accessories.iterrows():
                compatibility_list.append({'phone_product_id': phone['product_id'], 'accessory_product_id': acc['product_id']})
        else: # Lightning for older models
            lightning_accessories = products_df[products_df['compatible_tier'] == 'all_lightning']
            for _, acc in lightning_accessories.iterrows():
                compatibility_list.append({'phone_product_id': phone['product_id'], 'accessory_product_id': acc['product_id']})
    
    accessory_compatibility_df = pd.DataFrame(compatibility_list).drop_duplicates()
    
    return accessory_inventory_df, accessory_compatibility_df

# --- MAIN ORCHESTRATION FUNCTION ---

def main():
    """Main function to orchestrate the data generation and save files to a subdirectory."""
    print("--- Starting Synthetic Data Generation (v4) ---")
    
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
        print(f"Created output directory: {OUTPUT_DIRECTORY}/")
    
    stores_df = generate_stores()
    products_df = generate_products()
    inventory_units_df = generate_inventory_units(products_df)
    transactions_df = generate_transactions(inventory_units_df, products_df)
    accessory_inventory_df, accessory_compatibility_df = generate_accessory_data(products_df, stores_df)
    
    print(f"\nSaving dataframes to '{OUTPUT_DIRECTORY}/' directory...")
    try:
        file_paths = {
            "stores.csv": stores_df,
            "products.csv": products_df,
            "inventory_units.csv": inventory_units_df,
            "transactions.csv": transactions_df,
            "accessory_inventory.csv": accessory_inventory_df,
            "accessory_compatibility.csv": accessory_compatibility_df
        }
        
        for filename, df in file_paths.items():
            full_path = os.path.join(OUTPUT_DIRECTORY, filename)
            df.to_csv(full_path, index=False)
            print(f"- Successfully saved {full_path}")
            
    except Exception as e:
        print(f"An error occurred while saving files: {e}")

    print("\n--- Data Generation Complete ---")


if __name__ == "__main__":
    main()