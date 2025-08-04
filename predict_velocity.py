import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime, timedelta

class VelocityPredictor:
    """Loads and uses the pre-trained sales velocity model."""
    def __init__(self, model_path='models/velocity_model.joblib', products_path='data/products.csv', inventory_path='data/inventory_units.csv', transactions_path='data/transactions.csv'):
        try:
            self.pipeline = joblib.load(model_path)
            self.products_df = pd.read_csv(products_path)
            inventory = pd.read_csv(inventory_path)
            transactions = pd.read_csv(transactions_path)
        except FileNotFoundError as e:
            raise SystemExit(f"ERROR: {e}. Please run 'train_velocity_model.py' first.")

        # Prepare reference data
        self.products_df = self.products_df[self.products_df['product_type'] == 'Used Phone'].copy()
        self.products_df['base_model'] = self.products_df['model_name'].str.extract(r'(iPhone \d{1,2}(?: Pro| Plus| Pro Max| Mini)?)')[0]
        for col in ['release_date']:
            self.products_df[col] = pd.to_datetime(self.products_df[col], errors='coerce')
            
        # Create historical sales reference needed for market demand calculation
        sold_units = pd.merge(inventory[inventory['status'] == 'Sold'], transactions[['unit_id', 'transaction_date']], on='unit_id')
        sold_units['transaction_date'] = pd.to_datetime(sold_units['transaction_date'])
        self.sales_history = pd.merge(sold_units, self.products_df[['product_id', 'base_model']], on='product_id')

    def _get_live_market_demand(self, base_model: str, appraisal_date: datetime):
        """Calculates market demand based on the loaded sales history."""
        demand = self.sales_history[
            (self.sales_history['base_model'] == base_model) &
            (self.sales_history['transaction_date'] < appraisal_date) &
            (self.sales_history['transaction_date'] >= appraisal_date - timedelta(days=14))
        ].shape[0]

        if demand == 0:
            match = re.search(r'(\d+)', base_model)
            if match:
                predecessor = base_model.replace(str(match.group(1)), str(int(match.group(1)) - 1))
                if predecessor in self.products_df['base_model'].unique():
                    pred_sales = self.sales_history[
                        (self.sales_history['base_model'] == predecessor) &
                        (self.sales_history['transaction_date'] < appraisal_date) &
                        (self.sales_history['transaction_date'] >= appraisal_date - timedelta(days=90))
                    ]
                    if not pred_sales.empty:
                        demand = int(np.ceil(pred_sales.shape[0] / (90 / 7.0)))
        return demand

    def predict(self, model_name: str, grade: str, appraisal_date_str: str):
        """Predicts the sales velocity for a single phone."""
        appraisal_date = pd.to_datetime(appraisal_date_str)
        
        phone_details = self.products_df[self.products_df['model_name'] == model_name]
        if phone_details.empty:
            return {"error": f"Model '{model_name}' not found in products catalog."}
        
        phone_details = phone_details.iloc[0]

        # Engineer features for the input
        features = {
            'grade': grade,
            'storage_gb': phone_details['storage_gb'],
            'model_tier': phone_details['model_tier'],
            'original_msrp': phone_details['original_msrp'],
            'days_since_model_release': (appraisal_date - phone_details['release_date']).days,
            'market_demand': self._get_live_market_demand(phone_details['base_model'], appraisal_date)
        }
        features_df = pd.DataFrame([features])

        # Predict using the loaded pipeline
        prediction = self.pipeline.predict(features_df)[0]
            
        return {
            "model_appraised": model_name,
            "grade": grade,
            "predicted_sales_velocity": prediction
        }

if __name__ == "__main__":
    print("--- Sales Velocity Prediction Tool ---")
    predictor = VelocityPredictor()
    
    # Test a cold-start item
    result = predictor.predict(
        model_name='iPhone 11 64GB',
        grade='D',
        appraisal_date_str='2025-10-15'
    )
    
    print("\nAppraisal Result:")
    print(result)
