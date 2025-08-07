#
# Advanced Sales Velocity Prediction Model
#
# This script builds a sophisticated classification model. Its predictions
# for a specific item (e.g., a Grade 'A' iPhone 11) are based on the
# historical sales velocity of other items of the exact same model and grade.
#

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import numpy as np

def load_and_prepare_data(product_file, inventory_file, transaction_file):
    """
    Loads, merges, and prepares the data for model training.
    """
    products_df = pd.read_csv(product_file)
    inventory_units_df = pd.read_csv(inventory_file)
    transactions_df = pd.read_csv(transaction_file)

    inventory_units_df['acquisition_date'] = pd.to_datetime(inventory_units_df['acquisition_date'])
    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])

    phone_transactions_df = transactions_df.dropna(subset=['unit_id'])
    merged_df = pd.merge(phone_transactions_df, inventory_units_df, on='unit_id', how='left', suffixes=('_trans', '_inv'))
    merged_df = pd.merge(merged_df, products_df, left_on='product_id_trans', right_on='product_id', how='left')

    return merged_df

def define_target_variable(df):
    """
    Calculates the time an item spent in stock and categorizes it.
    """
    df['days_in_stock'] = (df['transaction_date'] - df['acquisition_date']).dt.days
    df = df[df['days_in_stock'] >= 0]

    fast_threshold = df['days_in_stock'].quantile(0.33)
    medium_threshold = df['days_in_stock'].quantile(0.66)

    print("--- Defining Mover Categories ---")
    print(f"Fast Mover: Sold in <= {fast_threshold:.0f} days")
    print(f"Medium Mover: Sold between {fast_threshold:.0f} and {medium_threshold:.0f} days")
    print(f"Dead Stock: Took > {medium_threshold:.0f} days to sell")

    conditions = [
        df['days_in_stock'] <= fast_threshold,
        (df['days_in_stock'] > fast_threshold) & (df['days_in_stock'] <= medium_threshold)
    ]
    choices = ['Fast Mover', 'Medium Mover']
    df['mover_category'] = np.select(conditions, choices, default='Dead Stock')

    return df

def engineer_grade_specific_features(df):
    """
    Creates features based on sales volume and velocity that are specific
    to each product AND each grade.
    """
    df = df.sort_values('transaction_date').copy()
    df_indexed = df.set_index('transaction_date')

    # **NEW LOGIC:** Group by both product and grade for feature calculation
    grouping_cols = ['product_id_trans', 'grade']

    # --- Feature 1: Grade-specific sales count in last 120 days ---
    sales_counts = df_indexed.groupby(grouping_cols)['unit_id'].rolling('120D').count()
    sales_counts = sales_counts.reset_index(name='grade_specific_sales_last_120d')
    df = pd.merge(df, sales_counts, on=['transaction_date', 'product_id_trans', 'grade'], how='left')

    # --- Feature 2: Grade-specific average days to sell in last 120 days ---
    avg_days = df_indexed.groupby(grouping_cols)['days_in_stock'].rolling('120D').mean()
    avg_days = avg_days.reset_index(name='grade_specific_avg_days_last_120d')
    df = pd.merge(df, avg_days, on=['transaction_date', 'product_id_trans', 'grade'], how='left')
    
    # Shift features to prevent data leakage (use data *before* the current sale)
    df['grade_specific_sales_last_120d'] = df.groupby(grouping_cols)['grade_specific_sales_last_120d'].shift(1)
    df['grade_specific_avg_days_last_120d'] = df.groupby(grouping_cols)['grade_specific_avg_days_last_120d'].shift(1)
    
    df.fillna(0, inplace=True)

    return df

def train_advanced_model(df):
    """
    Trains and evaluates the advanced classification model.
    """
    # ** The new, more sophisticated feature set **
    features = [
        'model_tier', 'storage_gb', 'original_msrp',
        'grade_specific_sales_last_120d',
        'grade_specific_avg_days_last_120d'
    ]
    target = 'mover_category'

    model_df = df.dropna(subset=features + [target])

    X = model_df[features]
    y = model_df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nTraining advanced model on {len(X_train)} samples...")

    gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gbc.fit(X_train, y_train)

    print("\nModel training complete. Performance on test data:")
    y_pred = gbc.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    return gbc, features

def predict_with_advanced_model(product_id, grade, model, full_df, features):
    """
    Generates a sales velocity report for a specific item using grade-specific features.
    """
    try:
        product_details = full_df[full_df['product_id'] == product_id].iloc[0]
        product_name = product_details['model_name']
    except IndexError:
        print(f"Error: Product ID {product_id} not found in the dataset.")
        return

    print(f"--- Advanced Prediction for: {product_name} (Grade: '{grade}') ---")

    # --- Get the most recent feature values for this specific product and grade ---
    item_history = full_df[
        (full_df['product_id'] == product_id) & 
        (full_df['grade'] == grade)
    ].sort_values('transaction_date')
    
    if item_history.empty:
        # If no history, assume 0 for feature values
        sales_count_recent = 0
        avg_days_recent = 0
    else:
        latest_stats = item_history.iloc[-1]
        sales_count_recent = latest_stats['grade_specific_sales_last_120d']
        avg_days_recent = latest_stats['grade_specific_avg_days_last_120d']

    # --- Model Prediction ---
    prediction_data = pd.DataFrame([{
        'model_tier': product_details['model_tier'],
        'storage_gb': product_details['storage_gb'],
        'original_msrp': product_details['original_msrp'],
        'grade_specific_sales_last_120d': sales_count_recent,
        'grade_specific_avg_days_last_120d': avg_days_recent
    }])
    prediction_data = prediction_data[features]

    predicted_category = model.predict(prediction_data)[0]
    print(f"Model Prediction: This item is likely to be a '{predicted_category}'.")

    # --- Explanation ---
    print("\nReasoning based on your historical data:")
    print(f"  - In the 120 days before the last sale of a Grade '{grade}' unit, you sold {int(sales_count_recent)} units of this specific item.")
    print(f"  - In that same period, the average time to sell them was {avg_days_recent:.1f} days.")


if __name__ == '__main__':
    # --- 1. Load and Prepare Data ---
    master_df = load_and_prepare_data(
        product_file='data/products.csv',
        inventory_file='data/inventory_units.csv',
        transaction_file='data/transactions.csv'
    )
    labeled_df = define_target_variable(master_df)

    # --- 2. Engineer Grade-Specific Features and Train Model ---
    featured_df = engineer_grade_specific_features(labeled_df)
    advanced_model, feature_list = train_advanced_model(featured_df)

    # --- 3. Run a Prediction for a Specific Item ---
    print("\n" + "="*60)
    print("      Advanced Prediction Explanations")
    print("="*60)

    # Example 1: Check the forecast for a Grade 'A' iPhone 13 128GB
    predict_with_advanced_model(
        product_id=25,
        grade='A',
        model=advanced_model,
        full_df=featured_df,
        features=feature_list
    )

    print("\n" + "-"*60 + "\n")

    # Example 2: Check the forecast for a Grade 'C' of the same model
    # This prediction will be different from Grade 'A' because it uses different feature values
    predict_with_advanced_model(
        product_id=69,
        grade='C',
        model=advanced_model,
        full_df=featured_df,
        features=feature_list
    )