import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import os
from sqlalchemy import create_engine
import urllib
from database_utils import SERVER_NAME as DB_SERVER_NAME, DATABASE_NAME as DB_NAME

START_DATE = datetime(2020, 8, 5)
END_DATE = datetime.now()
NUM_PHONE_UNITS = 10000
NUM_ACCESSORY_SALES = 10000
PHYSICAL_STORE_IDS = [1, 2, 3]
ALL_STORE_IDS = [1, 2, 3, 4]
fake = Faker()

def calculate_realistic_acquisition_price(
    product_details: pd.Series, acquisition_date: datetime.date, grade: str
) -> float:
    """Calculates a realistic acquisition price using user-defined calibrated depreciation."""
    base_price = product_details['original_msrp']
    used_market_value = base_price * 0.90
    days_old = max(0, (acquisition_date - product_details['release_date'].date()).days)
    daily_decay_rate = 0.0002
    used_market_value *= ((1 - daily_decay_rate) ** days_old)
    successor_release_date = product_details['successor_release_date']
    if pd.notna(successor_release_date) and acquisition_date > successor_release_date.date():
        used_market_value *= 0.80
    grade_acquisition_multipliers = {'A': 0.94, 'B': 0.78, 'C': 0.65, 'D': 0.40}
    acquisition_price_base = used_market_value * grade_acquisition_multipliers.get(grade, 0.5)
    final_price = acquisition_price_base * (1 + random.uniform(-0.03, 0.03))
    return round(max(0, final_price), 2)

def calculate_dynamic_max_margin(
    tier: float, grade: str, min_tier: float, max_tier: float
) -> float:
    """Calculates the maximum potential profit margin using a stable, subtractive penalty system."""
    newest_tier_margin = 1.12
    oldest_tier_margin = 1.05
    if max_tier == min_tier:
        normalized_tier = 1.0
    else:
        normalized_tier = (tier - min_tier) / (max_tier - min_tier)
    tier_based_margin = oldest_tier_margin + normalized_tier * (newest_tier_margin - oldest_tier_margin)
    grade_margin_penalties = {'A': 0.00, 'B': 0.012, 'C': 0.03, 'D': 0.5}
    penalty = grade_margin_penalties.get(grade, 0.07)
    final_base_margin = tier_based_margin - penalty
    return final_base_margin

def calculate_dynamic_days_on_market(
    tier: float, grade: str, min_tier: float, max_tier: float
) -> int:
    """Calculates a realistic selling time based on the phone's age and condition."""
    newest_model_params = {'mode': 15, 'high': 90}
    oldest_model_params = {'mode': 120, 'high': 365}
    if max_tier == min_tier:
        normalized_tier = 1.0
    else:
        normalized_tier = (tier - min_tier) / (max_tier - min_tier)
    mode = oldest_model_params['mode'] - normalized_tier * (oldest_model_params['mode'] - newest_model_params['mode'])
    high = oldest_model_params['high'] - normalized_tier * (oldest_model_params['high'] - newest_model_params['high'])
    grade_sluggishness = {'A': 1.0, 'B': 1.5, 'C': 2.0, 'D': 2.5}
    sluggishness_multiplier = grade_sluggishness.get(grade, 2.0)
    final_mode = min(mode * sluggishness_multiplier, high * sluggishness_multiplier)
    final_high = high * sluggishness_multiplier
    return int(random.triangular(low=1, high=final_high, mode=final_mode))

def generate_stores() -> pd.DataFrame:
    """Generates the stores table."""
    print("Generating stores table...")
    stores_data = [{'store_id': 1, 'store_name': 'Westfield Mall', 'location_type': 'Physical Store', 'city': 'Los Angeles'}, {'store_id': 2, 'store_name': 'Beverly Center', 'location_type': 'Physical Store', 'city': 'Los Angeles'}, {'store_id': 3, 'store_name': 'The Grove', 'location_type': 'Physical Store', 'city': 'Los Angeles'}, {'store_id': 4, 'store_name': 'Online Store', 'location_type': 'Online', 'city': 'N/A'}]
    return pd.DataFrame(stores_data)

def generate_products() -> pd.DataFrame:
    """Generates the master product catalog with an expanded accessory list."""
    print("Generating products table with expanded accessories...")
    phone_data = [

        {'model_name': 'iPhone 11 64GB', 'model_tier': 11.0, 'storage_gb': 64, 'original_msrp': 699, 'release_date': '2019-09-20', 'successor_release_date': '2020-10-23', 'form_factor': '11', 'connector': 'lightning'},
        {'model_name': 'iPhone 11 128GB', 'model_tier': 11.0, 'storage_gb': 128, 'original_msrp': 749, 'release_date': '2019-09-20', 'successor_release_date': '2020-10-23', 'form_factor': '11', 'connector': 'lightning'},
        {'model_name': 'iPhone 11 256GB', 'model_tier': 11.0, 'storage_gb': 256, 'original_msrp': 849, 'release_date': '2019-09-20', 'successor_release_date': '2020-10-23', 'form_factor': '11', 'connector': 'lightning'},
        {'model_name': 'iPhone 11 Pro 64GB', 'model_tier': 11.5, 'storage_gb': 64, 'original_msrp': 999, 'release_date': '2019-09-20', 'successor_release_date': '2020-10-23', 'form_factor': '11_pro', 'connector': 'lightning'},
        {'model_name': 'iPhone 11 Pro 256GB', 'model_tier': 11.5, 'storage_gb': 256, 'original_msrp': 1149, 'release_date': '2019-09-20', 'successor_release_date': '2020-10-23', 'form_factor': '11_pro', 'connector': 'lightning'},
        {'model_name': 'iPhone 11 Pro 512GB', 'model_tier': 11.5, 'storage_gb': 512, 'original_msrp': 1349, 'release_date': '2019-09-20', 'successor_release_date': '2020-10-23', 'form_factor': '11_pro', 'connector': 'lightning'},
        {'model_name': 'iPhone 11 Pro Max 64GB', 'model_tier': 11.7, 'storage_gb': 64, 'original_msrp': 1099, 'release_date': '2019-09-20', 'successor_release_date': '2020-10-23', 'form_factor': '11_pro_max', 'connector': 'lightning'},
        {'model_name': 'iPhone 11 Pro Max 256GB', 'model_tier': 11.7, 'storage_gb': 256, 'original_msrp': 1249, 'release_date': '2019-09-20', 'successor_release_date': '2020-10-23', 'form_factor': '11_pro_max', 'connector': 'lightning'},
        {'model_name': 'iPhone 11 Pro Max 512GB', 'model_tier': 11.7, 'storage_gb': 512, 'original_msrp': 1449, 'release_date': '2019-09-20', 'successor_release_date': '2020-10-23', 'form_factor': '11_pro_max', 'connector': 'lightning'},

        {'model_name': 'iPhone 12 Mini 64GB', 'model_tier': 12.2, 'storage_gb': 64, 'original_msrp': 599, 'release_date': '2020-11-13', 'successor_release_date': '2021-09-24', 'form_factor': '12_mini', 'connector': 'lightning'},
        {'model_name': 'iPhone 12 Mini 128GB', 'model_tier': 12.2, 'storage_gb': 128, 'original_msrp': 749, 'release_date': '2020-11-13', 'successor_release_date': '2021-09-24', 'form_factor': '12_mini', 'connector': 'lightning'},
        {'model_name': 'iPhone 12 Mini 256GB', 'model_tier': 12.2, 'storage_gb': 256, 'original_msrp': 849, 'release_date': '2020-11-13', 'successor_release_date': '2021-09-24', 'form_factor': '12_mini', 'connector': 'lightning'},
        {'model_name': 'iPhone 12 64GB', 'model_tier': 12.0, 'storage_gb': 64, 'original_msrp': 799, 'release_date': '2020-10-23', 'successor_release_date': '2021-09-24', 'form_factor': '12_12pro', 'connector': 'lightning'},
        {'model_name': 'iPhone 12 128GB', 'model_tier': 12.0, 'storage_gb': 128, 'original_msrp': 849, 'release_date': '2020-10-23', 'successor_release_date': '2021-09-24', 'form_factor': '12_12pro', 'connector': 'lightning'},
        {'model_name': 'iPhone 12 256GB', 'model_tier': 12.0, 'storage_gb': 256, 'original_msrp': 949, 'release_date': '2020-10-23', 'successor_release_date': '2021-09-24', 'form_factor': '12_12pro', 'connector': 'lightning'},
        {'model_name': 'iPhone 12 Pro 128GB', 'model_tier': 12.5, 'storage_gb': 128, 'original_msrp': 999, 'release_date': '2020-10-23', 'successor_release_date': '2021-09-24', 'form_factor': '12_12pro', 'connector': 'lightning'},
        {'model_name': 'iPhone 12 Pro 256GB', 'model_tier': 12.5, 'storage_gb': 256, 'original_msrp': 1099, 'release_date': '2020-10-23', 'successor_release_date': '2021-09-24', 'form_factor': '12_12pro', 'connector': 'lightning'},
        {'model_name': 'iPhone 12 Pro 512GB', 'model_tier': 12.5, 'storage_gb': 512, 'original_msrp': 1299, 'release_date': '2020-10-23', 'successor_release_date': '2021-09-24', 'form_factor': '12_12pro', 'connector': 'lightning'},
        {'model_name': 'iPhone 12 Pro Max 128GB', 'model_tier': 12.7, 'storage_gb': 128, 'original_msrp': 1099, 'release_date': '2020-11-13', 'successor_release_date': '2021-09-24', 'form_factor': '12_pro_max', 'connector': 'lightning'},
        {'model_name': 'iPhone 12 Pro Max 256GB', 'model_tier': 12.7, 'storage_gb': 256, 'original_msrp': 1199, 'release_date': '2020-11-13', 'successor_release_date': '2021-09-24', 'form_factor': '12_pro_max', 'connector': 'lightning'},
        {'model_name': 'iPhone 12 Pro Max 512GB', 'model_tier': 12.7, 'storage_gb': 512, 'original_msrp': 1399, 'release_date': '2020-11-13', 'successor_release_date': '2021-09-24', 'form_factor': '12_pro_max', 'connector': 'lightning'},

        {'model_name': 'iPhone 13 Mini 128GB', 'model_tier': 13.2, 'storage_gb': 128, 'original_msrp': 699, 'release_date': '2021-09-24', 'successor_release_date': '2022-09-16', 'form_factor': '13_mini', 'connector': 'lightning'},
        {'model_name': 'iPhone 13 Mini 256GB', 'model_tier': 13.2, 'storage_gb': 256, 'original_msrp': 799, 'release_date': '2021-09-24', 'successor_release_date': '2022-09-16', 'form_factor': '13_mini', 'connector': 'lightning'},
        {'model_name': 'iPhone 13 Mini 512GB', 'model_tier': 13.2, 'storage_gb': 512, 'original_msrp': 999, 'release_date': '2021-09-24', 'successor_release_date': '2022-09-16', 'form_factor': '13_mini', 'connector': 'lightning'},
        {'model_name': 'iPhone 13 128GB', 'model_tier': 13.0, 'storage_gb': 128, 'original_msrp': 799, 'release_date': '2021-09-24', 'successor_release_date': '2022-09-16', 'form_factor': '13_13pro', 'connector': 'lightning'},
        {'model_name': 'iPhone 13 256GB', 'model_tier': 13.0, 'storage_gb': 256, 'original_msrp': 899, 'release_date': '2021-09-24', 'successor_release_date': '2022-09-16', 'form_factor': '13_13pro', 'connector': 'lightning'},
        {'model_name': 'iPhone 13 512GB', 'model_tier': 13.0, 'storage_gb': 512, 'original_msrp': 1099, 'release_date': '2021-09-24', 'successor_release_date': '2022-09-16', 'form_factor': '13_13pro', 'connector': 'lightning'},
        {'model_name': 'iPhone 13 Pro 128GB', 'model_tier': 13.5, 'storage_gb': 128, 'original_msrp': 999, 'release_date': '2021-09-24', 'successor_release_date': '2022-09-16', 'form_factor': '13_13pro', 'connector': 'lightning'},
        {'model_name': 'iPhone 13 Pro 256GB', 'model_tier': 13.5, 'storage_gb': 256, 'original_msrp': 1099, 'release_date': '2021-09-24', 'successor_release_date': '2022-09-16', 'form_factor': '13_13pro', 'connector': 'lightning'},
        {'model_name': 'iPhone 13 Pro 512GB', 'model_tier': 13.5, 'storage_gb': 512, 'original_msrp': 1299, 'release_date': '2021-09-24', 'successor_release_date': '2022-09-16', 'form_factor': '13_13pro', 'connector': 'lightning'},
        {'model_name': 'iPhone 13 Pro 1TB', 'model_tier': 13.5, 'storage_gb': 1024, 'original_msrp': 1499, 'release_date': '2021-09-24', 'successor_release_date': '2022-09-16', 'form_factor': '13_13pro', 'connector': 'lightning'},
        {'model_name': 'iPhone 13 Pro Max 128GB', 'model_tier': 13.7, 'storage_gb': 128, 'original_msrp': 1099, 'release_date': '2021-09-24', 'successor_release_date': '2022-09-16', 'form_factor': '13_pro_max', 'connector': 'lightning'},
        {'model_name': 'iPhone 13 Pro Max 256GB', 'model_tier': 13.7, 'storage_gb': 256, 'original_msrp': 1199, 'release_date': '2021-09-24', 'successor_release_date': '2022-09-16', 'form_factor': '13_pro_max', 'connector': 'lightning'},
        {'model_name': 'iPhone 13 Pro Max 512GB', 'model_tier': 13.7, 'storage_gb': 512, 'original_msrp': 1399, 'release_date': '2021-09-24', 'successor_release_date': '2022-09-16', 'form_factor': '13_pro_max', 'connector': 'lightning'},
        {'model_name': 'iPhone 13 Pro Max 1TB', 'model_tier': 13.7, 'storage_gb': 1024, 'original_msrp': 1499, 'release_date': '2021-09-24', 'successor_release_date': '2022-09-16', 'form_factor': '13_pro_max', 'connector': 'lightning'},

        {'model_name': 'iPhone 14 128GB', 'model_tier': 14.0, 'storage_gb': 128, 'original_msrp': 799, 'release_date': '2022-09-16', 'successor_release_date': '2023-09-22', 'form_factor': '14', 'connector': 'lightning'},
        {'model_name': 'iPhone 14 256GB', 'model_tier': 14.0, 'storage_gb': 256, 'original_msrp': 899, 'release_date': '2022-09-16', 'successor_release_date': '2023-09-22', 'form_factor': '14', 'connector': 'lightning'},
        {'model_name': 'iPhone 14 512GB', 'model_tier': 14.0, 'storage_gb': 512, 'original_msrp': 1099, 'release_date': '2022-09-16', 'successor_release_date': '2023-09-22', 'form_factor': '14', 'connector': 'lightning'},
        {'model_name': 'iPhone 14 Plus 128GB', 'model_tier': 14.2, 'storage_gb': 128, 'original_msrp': 899, 'release_date': '2022-10-07', 'successor_release_date': '2023-09-22', 'form_factor': '14_plus', 'connector': 'lightning'},
        {'model_name': 'iPhone 14 Plus 256GB', 'model_tier': 14.2, 'storage_gb': 256, 'original_msrp': 999, 'release_date': '2022-10-07', 'successor_release_date': '2023-09-22', 'form_factor': '14_plus', 'connector': 'lightning'},
        {'model_name': 'iPhone 14 Plus 512GB', 'model_tier': 14.2, 'storage_gb': 512, 'original_msrp': 1199, 'release_date': '2022-10-07', 'successor_release_date': '2023-09-22', 'form_factor': '14_plus', 'connector': 'lightning'},
        {'model_name': 'iPhone 14 Pro 128GB', 'model_tier': 14.5, 'storage_gb': 128, 'original_msrp': 1099, 'release_date': '2022-09-16', 'successor_release_date': '2023-09-22', 'form_factor': '14_pro', 'connector': 'lightning'},
        {'model_name': 'iPhone 14 Pro 256GB', 'model_tier': 14.5, 'storage_gb': 256, 'original_msrp': 1199, 'release_date': '2022-09-16', 'successor_release_date': '2023-09-22', 'form_factor': '14_pro', 'connector': 'lightning'},
        {'model_name': 'iPhone 14 Pro 512GB', 'model_tier': 14.5, 'storage_gb': 512, 'original_msrp': 1399, 'release_date': '2022-09-16', 'successor_release_date': '2023-09-22', 'form_factor': '14_pro', 'connector': 'lightning'},
        {'model_name': 'iPhone 14 Pro 1TB', 'model_tier': 14.5, 'storage_gb': 1024, 'original_msrp': 1599, 'release_date': '2022-09-16', 'successor_release_date': '2023-09-22', 'form_factor': '14_pro', 'connector': 'lightning'},
        {'model_name': 'iPhone 14 Pro Max 128GB', 'model_tier': 14.7, 'storage_gb': 128, 'original_msrp': 1199, 'release_date': '2022-09-16', 'successor_release_date': '2023-09-22', 'form_factor': '14_pro_max', 'connector': 'lightning'},
        {'model_name': 'iPhone 14 Pro Max 256GB', 'model_tier': 14.7, 'storage_gb': 256, 'original_msrp': 1299, 'release_date': '2022-09-16', 'successor_release_date': '2023-09-22', 'form_factor': '14_pro_max', 'connector': 'lightning'},
        {'model_name': 'iPhone 14 Pro Max 512GB', 'model_tier': 14.7, 'storage_gb': 512, 'original_msrp': 1499, 'release_date': '2022-09-16', 'successor_release_date': '2023-09-22', 'form_factor': '14_pro_max', 'connector': 'lightning'},
        {'model_name': 'iPhone 14 Pro Max 1TB', 'model_tier': 14.7, 'storage_gb': 1024, 'original_msrp': 1699, 'release_date': '2022-09-16', 'successor_release_date': '2023-09-22', 'form_factor': '14_pro_max', 'connector': 'lightning'},

        {'model_name': 'iPhone 15 256GB', 'model_tier': 15.0, 'storage_gb': 256, 'original_msrp': 899, 'release_date': '2023-09-22', 'successor_release_date': '2024-09-20', 'form_factor': '15', 'connector': 'usb-c'},
        {'model_name': 'iPhone 15 512GB', 'model_tier': 15.0, 'storage_gb': 512, 'original_msrp': 1099, 'release_date': '2023-09-22', 'successor_release_date': '2024-09-20', 'form_factor': '15', 'connector': 'usb-c'},
        {'model_name': 'iPhone 15 Plus 256GB', 'model_tier': 15.2, 'storage_gb': 256, 'original_msrp': 999, 'release_date': '2023-09-22', 'successor_release_date': '2024-09-20', 'form_factor': '15_plus', 'connector': 'usb-c'},
        {'model_name': 'iPhone 15 Pro 256GB', 'model_tier': 15.5, 'storage_gb': 256, 'original_msrp': 1199, 'release_date': '2023-09-22', 'successor_release_date': '2024-09-20', 'form_factor': '15_pro', 'connector': 'usb-c'},
        {'model_name': 'iPhone 15 Pro 512GB', 'model_tier': 15.5, 'storage_gb': 512, 'original_msrp': 1399, 'release_date': '2023-09-22', 'successor_release_date': '2024-09-20', 'form_factor': '15_pro', 'connector': 'usb-c'},
        {'model_name': 'iPhone 15 Pro 1TB', 'model_tier': 15.5, 'storage_gb': 1024, 'original_msrp': 1599, 'release_date': '2023-09-22', 'successor_release_date': '2024-09-20', 'form_factor': '15_pro', 'connector': 'usb-c'},
        {'model_name': 'iPhone 15 Pro Max 256GB', 'model_tier': 15.7, 'storage_gb': 256, 'original_msrp': 1399, 'release_date': '2023-09-22', 'successor_release_date': '2024-09-20', 'form_factor': '15_pro_max', 'connector': 'usb-c'},
        {'model_name': 'iPhone 15 Pro Max 512GB', 'model_tier': 15.7, 'storage_gb': 512, 'original_msrp': 1599, 'release_date': '2023-09-22', 'successor_release_date': '2024-09-20', 'form_factor': '15_pro_max', 'connector': 'usb-c'},
        {'model_name': 'iPhone 15 Pro Max 1TB', 'model_tier': 15.7, 'storage_gb': 1024, 'original_msrp': 1799, 'release_date': '2023-09-22', 'successor_release_date': '2024-09-20', 'form_factor': '15_pro_max', 'connector': 'usb-c'},

        {'model_name': 'iPhone 16 128GB', 'model_tier': 16.0, 'storage_gb': 128, 'original_msrp': 829, 'release_date': '2024-09-20', 'successor_release_date': None, 'form_factor': '16', 'connector': 'usb-c'},
        {'model_name': 'iPhone 16 256GB', 'model_tier': 16.0, 'storage_gb': 256, 'original_msrp': 899, 'release_date': '2024-09-20', 'successor_release_date': None, 'form_factor': '16', 'connector': 'usb-c'},
        {'model_name': 'iPhone 16 512GB', 'model_tier': 16.0, 'storage_gb': 512, 'original_msrp': 999, 'release_date': '2024-09-20', 'successor_release_date': None, 'form_factor': '16', 'connector': 'usb-c'},
        {'model_name': 'iPhone 16 Plus 128GB', 'model_tier': 16.2, 'storage_gb': 128, 'original_msrp': 929, 'release_date': '2024-09-20', 'successor_release_date': None, 'form_factor': '16_plus', 'connector': 'usb-c'},
        {'model_name': 'iPhone 16 Plus 256GB', 'model_tier': 16.2, 'storage_gb': 256, 'original_msrp': 999, 'release_date': '2024-09-20', 'successor_release_date': None, 'form_factor': '16_plus', 'connector': 'usb-c'},
        {'model_name': 'iPhone 16 Plus 512GB', 'model_tier': 16.2, 'storage_gb': 512, 'original_msrp': 1099, 'release_date': '2024-09-20', 'successor_release_date': None, 'form_factor': '16_plus', 'connector': 'usb-c'},
        {'model_name': 'iPhone 16 Pro 128GB', 'model_tier': 16.5, 'storage_gb': 128, 'original_msrp': 999, 'release_date': '2024-09-20', 'successor_release_date': None, 'form_factor': '16_pro', 'connector': 'usb-c'},
        {'model_name': 'iPhone 16 Pro 256GB', 'model_tier': 16.5, 'storage_gb': 256, 'original_msrp': 1099, 'release_date': '2024-09-20', 'successor_release_date': None, 'form_factor': '16_pro', 'connector': 'usb-c'},
        {'model_name': 'iPhone 16 Pro 512GB', 'model_tier': 16.5, 'storage_gb': 512, 'original_msrp': 1299, 'release_date': '2024-09-20', 'successor_release_date': None, 'form_factor': '16_pro', 'connector': 'usb-c'},
        {'model_name': 'iPhone 16 Pro 1TB', 'model_tier': 16.5, 'storage_gb': 1024, 'original_msrp': 1499, 'release_date': '2024-09-20', 'successor_release_date': None, 'form_factor': '16_pro', 'connector': 'usb-c'},
        {'model_name': 'iPhone 16 Pro Max 256GB', 'model_tier': 16.7, 'storage_gb': 256, 'original_msrp': 1199, 'release_date': '2024-09-20', 'successor_release_date': None, 'form_factor': '16_pro_max', 'connector': 'usb-c'},
        {'model_name': 'iPhone 16 Pro Max 512GB', 'model_tier': 16.7, 'storage_gb': 512, 'original_msrp': 1399, 'release_date': '2024-09-20', 'successor_release_date': None, 'form_factor': '16_pro_max', 'connector': 'usb-c'},
        {'model_name': 'iPhone 16 Pro Max 1TB', 'model_tier': 16.7, 'storage_gb': 1024, 'original_msrp': 1599, 'release_date': '2024-09-20', 'successor_release_date': None, 'form_factor': '16_pro_max', 'connector': 'usb-c'},
    ]
    for p in phone_data: p['product_type'] = 'Used Phone'
    
    accessory_data = []
    form_factors = {p['form_factor']: p['model_name'] for p in phone_data}
    for ff_code, ff_name in form_factors.items():
        accessory_data.append({'model_name': f'{ff_name} Silicone Case', 'original_msrp': 49, 'compatibility': {'form_factor': ff_code}})
        accessory_data.append({'model_name': f'{ff_name} Tempered Glass Screen Protector', 'original_msrp': 39, 'compatibility': {'form_factor': ff_code}})
    accessory_data.extend([{'model_name': '20W USB-C Power Adapter', 'original_msrp': 19, 'compatibility': {'type': 'universal'}}, {'model_name': 'MagSafe Charger', 'original_msrp': 39, 'compatibility': {'min_tier': 12.0}}, {'model_name': 'Lightning Cable (1m)', 'original_msrp': 19, 'compatibility': {'connector': 'lightning'}}, {'model_name': 'USB-C Cable (1m)', 'original_msrp': 19, 'compatibility': {'connector': 'usb-c'}}, {'model_name': 'AirPods Pro (2nd Gen)', 'original_msrp': 249, 'compatibility': {'type': 'universal'}}])
    for p in accessory_data: p['product_type'] = 'Accessory'

    products_list = phone_data + accessory_data
    df = pd.DataFrame(products_list)
    df.loc[df['product_type'] == 'Accessory', 'storage_gb'] = np.nan
    df['product_id'] = range(1, len(df) + 1)
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['successor_release_date'] = pd.to_datetime(df['successor_release_date'])
    return df

def generate_inventory_and_transactions(
    products_df: pd.DataFrame, accessory_compatibility: Dict[int, List[int]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates inventory and transaction data.
    Transaction IDs are assigned chronologically *after* all transactions are generated.
    """
    print("Starting integrated inventory and transaction simulation...")
    phone_products = products_df[products_df['product_type'] == 'Used Phone'].copy()
    min_tier = phone_products['model_tier'].min()
    max_tier = phone_products['model_tier'].max()
    phone_products.set_index('product_id', inplace=True)
    inventory_list, transactions_list = [], []
    total_days_in_history = (END_DATE - START_DATE).days
    acquisition_dates = [START_DATE + timedelta(days=random.randint(0, total_days_in_history)) for _ in range(NUM_PHONE_UNITS)]
    basket_id_counter = 1
    
    generated_accessory_sales = 0
    
    for i in range(NUM_PHONE_UNITS):
        unit_id = i + 1
        product_id = random.choice(phone_products.index)
        product_details = phone_products.loc[product_id]
        acquisition_date = acquisition_dates[i]
        while acquisition_date.date() < product_details['release_date'].date():
            acquisition_date = START_DATE + timedelta(days=random.randint(0, total_days_in_history))
        grade = random.choices(['A', 'B', 'C', 'D'], weights=[0.4, 0.35, 0.2, 0.05], k=1)[0]
        age_at_acquisition_days = (acquisition_date.date() - product_details['release_date'].date()).days
        acquisition_price = calculate_realistic_acquisition_price(product_details, acquisition_date.date(), grade)
        days_on_market = calculate_dynamic_days_on_market(tier=product_details['model_tier'], grade=grade, min_tier=min_tier, max_tier=max_tier)
        potential_sale_date = acquisition_date + timedelta(days=days_on_market)
        unit_record = {'unit_id': unit_id, 'product_id': product_id, 'store_id': random.choice(PHYSICAL_STORE_IDS), 'acquisition_date': acquisition_date.date(), 'grade': grade, 'age_at_acquisition_days': age_at_acquisition_days, 'acquisition_price': acquisition_price}
        if potential_sale_date <= END_DATE:
            unit_record['status'] = 'Sold'
            max_profit_margin = calculate_dynamic_max_margin(tier=product_details['model_tier'], grade=grade, min_tier=min_tier, max_tier=max_tier)
            margin_decay_per_day = 0.0005
            floor_margin = 0.90
            dynamic_profit_margin = max_profit_margin - (days_on_market * margin_decay_per_day)
            final_profit_margin = max(floor_margin, dynamic_profit_margin)
            calculated_sale_price = acquisition_price * final_profit_margin
            final_sale_price = min(calculated_sale_price, product_details['original_msrp'])
            
            phone_t_record = {'transaction_id': basket_id_counter, 'unit_id': unit_id, 'product_id': product_id, 'store_id': random.choice(ALL_STORE_IDS), 'transaction_date': potential_sale_date.date(), 'age_at_sale_days': (potential_sale_date.date() - product_details['release_date'].date()).days, 'final_sale_price': round(final_sale_price, 2), 'transaction_type': 'Sale'}
            transactions_list.append(phone_t_record)
            
            if random.random() < 0.40:
                num_accessories = random.randint(1, 2)
                compatible_acc_ids = accessory_compatibility.get(product_id, [])
                if compatible_acc_ids:
                    accessories_to_add = random.sample(compatible_acc_ids, k=min(num_accessories, len(compatible_acc_ids)))
                    for acc_id in accessories_to_add:
                        acc_details = products_df.loc[products_df['product_id'] == acc_id].iloc[0]
                        transactions_list.append({'transaction_id': basket_id_counter, 'unit_id': np.nan, 'product_id': acc_id, 'store_id': phone_t_record['store_id'], 'transaction_date': phone_t_record['transaction_date'], 'age_at_sale_days': np.nan, 'final_sale_price': round(acc_details['original_msrp'] * 1.20, 2), 'transaction_type': 'Sale'})
                        generated_accessory_sales += 1
            basket_id_counter += 1
        else:
            unit_record['status'] = 'In Stock'
        inventory_list.append(unit_record)

    num_standalone_accessories = NUM_ACCESSORY_SALES - generated_accessory_sales
    print(f"Generating {num_standalone_accessories} standalone accessory sales...")
    accessory_products = products_df[products_df['product_type'] == 'Accessory']
    for _ in range(num_standalone_accessories):
        product_details = accessory_products.sample(1).iloc[0]
        transactions_list.append({'transaction_id': basket_id_counter, 'unit_id': np.nan, 'product_id': product_details['product_id'], 'store_id': random.choice(ALL_STORE_IDS), 'transaction_date': (START_DATE + timedelta(days=random.randint(0, total_days_in_history))).date(), 'age_at_sale_days': np.nan, 'final_sale_price': round(product_details['original_msrp'] * 1.20, 2), 'transaction_type': 'Sale'})
        basket_id_counter += 1
        
    if not transactions_list:
        return pd.DataFrame(inventory_list), pd.DataFrame()
        
    transactions_df = pd.DataFrame(transactions_list)
    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
    transactions_df.sort_values(by=['transaction_date', 'transaction_id'], inplace=True)
    chronological_basket_ids = transactions_df['transaction_id'].unique()
    basket_to_new_id_map = {old_id: new_id for new_id, old_id in enumerate(chronological_basket_ids, 1)}
    transactions_df['transaction_id'] = transactions_df['transaction_id'].map(basket_to_new_id_map)
    transactions_df['transaction_date'] = transactions_df['transaction_date'].dt.date

    return pd.DataFrame(inventory_list), transactions_df

def generate_accessory_data(
    products_df: pd.DataFrame, stores_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, List[int]]]:
    """Generates accessory inventory and a detailed compatibility map."""
    print("Generating accessory inventory and compatibility tables...")
    accessory_inventory_list, compatibility_list = [], []
    phone_to_acc_map = {}
    accessories = products_df[products_df['product_type'] == 'Accessory']
    phones = products_df[products_df['product_type'] == 'Used Phone']
    physical_stores = stores_df[stores_df['location_type'] == 'Physical Store']
    for _, store in physical_stores.iterrows():
        for _, acc in accessories.iterrows():
            accessory_inventory_list.append({'store_id': store['store_id'], 'product_id': acc['product_id'], 'quantity': random.randint(10, 100)})
    for _, acc in accessories.iterrows():
        rules = acc['compatibility']
        for _, phone in phones.iterrows():
            is_compatible = False
            if rules.get('type') == 'universal': is_compatible = True
            elif 'form_factor' in rules and rules['form_factor'] == phone['form_factor']: is_compatible = True
            elif 'connector' in rules and rules['connector'] == phone['connector']: is_compatible = True
            elif 'min_tier' in rules and phone['model_tier'] >= rules['min_tier']: is_compatible = True
            if is_compatible:
                phone_id = phone['product_id']
                acc_id = acc['product_id']
                compatibility_list.append({'phone_product_id': phone_id, 'accessory_product_id': acc_id})
                if phone_id not in phone_to_acc_map:
                    phone_to_acc_map[phone_id] = []
                phone_to_acc_map[phone_id].append(acc_id)
    return pd.DataFrame(accessory_inventory_list), pd.DataFrame(compatibility_list), phone_to_acc_map

def load_dataframes_to_db(dataframe_dict: Dict[str, pd.DataFrame]):
    """
    Loads a dictionary of pandas DataFrames into a new SQL Server database.
    Each DataFrame will become a table in the database.
    """
    try:

        params = urllib.parse.quote_plus(
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={DB_SERVER_NAME};"
            f"DATABASE={DB_NAME};"
            f"Trusted_Connection=yes;"
        )
        connection_string = f"mssql+pyodbc:///?odbc_connect={params}"
        engine = create_engine(connection_string)

        with engine.connect() as connection:
            print(f"\nSuccessfully connected to SQL Server: {DB_SERVER_NAME}, Database: {DB_NAME}")
            print(f"Found {len(dataframe_dict)} dataframes to load:")
            
            for table_name, df in dataframe_dict.items():
                print(f"  - Loading DataFrame into table '{table_name}'...")
                df.to_sql(table_name, engine, if_exists='replace', index=False)
                print(f"    ...Done. Table '{table_name}' created with {len(df)} rows.")

            print("\nDatabase creation and data loading complete.")
            print("You can now verify the tables in SQL Server Management Studio (SSMS).")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("\nPlease check the following:")
        print("1. Is the SQL Server instance running?")
        print(f"2. Is the SERVER_NAME ('{DB_SERVER_NAME}') and DATABASE_NAME ('{DB_NAME}') in the script correct?")
        print("3. Have you installed the required libraries (`pip install sqlalchemy pyodbc pandas numpy faker`)?")
        print("4. Is the 'ODBC Driver 17 for SQL Server' installed?")

def main():
    """Main function to generate data and load it directly into SQL Server."""
    print("--- Starting Synthetic Data Generation and Database Loading ---")

    print("\n--- Step 1: Generating Data ---")
    stores_df = generate_stores()
    products_df = generate_products()
    accessory_inventory_df, accessory_compatibility_df, phone_to_acc_map = generate_accessory_data(products_df, stores_df)
    inventory_units_df, transactions_df = generate_inventory_and_transactions(products_df, phone_to_acc_map)
    print("--- Data Generation Complete ---")

    dataframe_dict = {
        "stores": stores_df,
        "products": products_df,
        "inventory_units": inventory_units_df,
        "transactions": transactions_df,
        "accessory_inventory": accessory_inventory_df,
        "accessory_compatibility": accessory_compatibility_df
    }
    
    print("\n--- Step 2: Loading Data into SQL Server ---")
    load_dataframes_to_db(dataframe_dict)
    
    print("\n--- Full Process Complete ---")

if __name__ == "__main__":
    main()