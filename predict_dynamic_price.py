import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime, timedelta

class DynamicPricePredictor:
    """Loads and uses the pre-trained dynamic selling price model."""
    def __init__(self, model_path='models/dynamic_selling_price_pipeline.joblib', products_path='data/products.csv', inventory_path='data/inventory_units.csv', transactions_path='data/transactions.csv'):
        try:
            self.pipeline = joblib.load(model_path)
            self.products_df = pd.read_csv(products_path)
            self.inventory_df = pd.read_csv(inventory_path)
            transactions = pd.read_csv(transactions_path)
        except FileNotFoundError as e:
            raise SystemExit(f"ERROR: {e}. Please run 'train_dynamic_price_model.py' first.")

        # Prepare reference data
        self._prepare_reference_data(transactions)

    def _prepare_reference_data(self, transactions):
        """Helper to prep all loaded dataframes for use in prediction."""
        # Products
        self.products_df = self.products_df[self.products_df['product_type'] == 'Used Phone'].copy()
        self.products_df['base_model'] = self.products_df['model_name'].str.extract(r'(iPhone \d{1,2}(?: Pro| Plus| Pro Max| Mini)?)')[0]
        for col in ['release_date', 'successor_release_date']:
            self.products_df[col] = pd.to_datetime(self.products_df[col], errors='coerce')
        
        # Inventory
        self.inventory_df['acquisition_date'] = pd.to_datetime(self.inventory_df['acquisition_date'])
        
        # Historical Sales for market demand calculation
        transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
        sold_inventory_info = pd.merge(self.inventory_df, self.products_df[['product_id', 'base_model']], on='product_id')
        self.sales_history = pd.merge(transactions, sold_inventory_info[['unit_id', 'base_model']], on='unit_id')

    def _get_live_market_demand(self, base_model: str, prediction_date: datetime):
        """Calculates market demand based on the loaded sales history up to the prediction date."""
        def get_predecessor(bm):
            match = re.search(r'(\d+)', bm)
            if not match: return None
            return bm.replace(str(match.group(1)), str(int(match.group(1)) - 1))

        demand = self.sales_history[
            (self.sales_history['base_model'] == base_model) &
            (self.sales_history['transaction_date'] < prediction_date) &
            (self.sales_history['transaction_date'] >= prediction_date - timedelta(days=7))
        ].shape[0]

        if demand == 0:
            predecessor = get_predecessor(base_model)
            if predecessor and predecessor in self.products_df['base_model'].unique():
                pred_sales = self.sales_history[
                    (self.sales_history['base_model'] == predecessor) &
                    (self.sales_history['transaction_date'] < prediction_date) &
                    (self.sales_history['transaction_date'] >= prediction_date - timedelta(days=7))
                ]
                demand = pred_sales.shape[0]
        return demand

    def predict(self, unit_id: int, prediction_date_str: str):
        """Predicts the dynamic selling price range for a specific inventory unit."""
        prediction_date = pd.to_datetime(prediction_date_str)
        
        # 1. Get unit details
        unit_info = self.inventory_df[self.inventory_df['unit_id'] == unit_id]
        if unit_info.empty:
            return {"error": f"Unit ID {unit_id} not found in inventory."}
        
        # 2. Merge with product details
        full_details = pd.merge(unit_info, self.products_df, on='product_id').iloc[0]

        # 3. Engineer features for the specific unit and date
        features = {
            'original_msrp': full_details['original_msrp'],
            'storage_gb': full_details['storage_gb'],
            'grade': full_details['grade'],
            'model_tier': full_details['model_tier'],
            'base_model': full_details['base_model'],
            'acquisition_price': full_details['acquisition_price'],
            'days_in_inventory': (prediction_date - full_details['acquisition_date']).days,
            'days_since_model_release': (prediction_date - full_details['release_date']).days,
            'days_since_successor_release': (prediction_date - full_details['successor_release_date']).days if pd.notna(full_details['successor_release_date']) else -999,
            'market_demand_7_days': self._get_live_market_demand(full_details['base_model'], prediction_date)
        }
        features_df = pd.DataFrame([features])

        # 4. Predict using the loaded pipeline
        preprocessor = self.pipeline['preprocessor']
        models = self.pipeline['models']
        features_processed = preprocessor.transform(features_df)
        
        price_prediction = {}
        for name, model in models.items():
            price_prediction[name] = round(model.predict(features_processed)[0], 2)
            
        return {
            "unit_id": unit_id,
            "model_name": full_details['model_name'],
            "status": full_details['status'],
            "days_in_inventory": features['days_in_inventory'],
            "predicted_selling_price_range": price_prediction
        }

if __name__ == "__main__":
    print("--- Dynamic Selling Price Prediction Tool ---")
    predictor = DynamicPricePredictor()
    
    # Find an item that is currently 'In Stock' to predict its price
    in_stock_inventory = pd.read_csv('data/inventory_units.csv')
    item_to_price = in_stock_inventory[in_stock_inventory['status'] == 'In Stock'].iloc[0]
    unit_to_predict = item_to_price['unit_id']
    
    print(f"\nPredicting price for Unit ID: {unit_to_predict} (Acquired on {item_to_price['acquisition_date']})")

    # Test Case 1: Price it shortly after acquisition
    result1 = predictor.predict(
        unit_id=unit_to_predict,
        prediction_date_str='2024-11-02' # Assuming this is soon after acquisition
    )
    print("\nPrediction (Soon after acquisition):")
    print(result1)
    
    # Test Case 2: Price it much later
    result2 = predictor.predict(
        unit_id=unit_to_predict,
        prediction_date_str='2025-03-20' # Several months later
    )
    print("\nPrediction (Several months later):")
    print(result2)
    print("\nNote: The price is expected to drop as 'days_in_inventory' increases.")
