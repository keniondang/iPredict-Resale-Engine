import pandas as pd
import joblib
from datetime import datetime

class PricePredictor:
    """Loads and uses the pre-trained price prediction model."""
    def __init__(self, model_path='models/price_model.joblib', products_path='data/products.csv'):
        try:
            self.pipeline = joblib.load(model_path)
            self.products_df = pd.read_csv(products_path)
        except FileNotFoundError as e:
            raise SystemExit(f"ERROR: {e}. Please run 'train_price_model.py' first.")

        # Prepare reference data
        self.products_df = self.products_df[self.products_df['product_type'] == 'Used Phone'].copy()
        self.products_df['base_model'] = self.products_df['model_name'].str.extract(r'(iPhone \d{1,2}(?: Pro| Plus| Pro Max| Mini)?)')[0]
        for col in ['release_date', 'successor_release_date']:
            self.products_df[col] = pd.to_datetime(self.products_df[col], errors='coerce')

    def predict(self, model_name: str, grade: str, appraisal_date_str: str):
        """Predicts the price range for a single phone."""
        appraisal_date = pd.to_datetime(appraisal_date_str)
        
        phone_details = self.products_df[self.products_df['model_name'] == model_name]
        if phone_details.empty:
            return {"error": f"Model '{model_name}' not found in products catalog."}
        
        phone_details = phone_details.iloc[0]

        # Engineer features for the input
        features = {
            'base_model': phone_details['base_model'],
            'grade': grade,
            'storage_gb': phone_details['storage_gb'],
            'model_tier': phone_details['model_tier'],
            'original_msrp': phone_details['original_msrp'],
            'days_since_model_release': (appraisal_date - phone_details['release_date']).days,
            'days_since_successor_release': (appraisal_date - phone_details['successor_release_date']).days if pd.notna(phone_details['successor_release_date']) else -999
        }
        features_df = pd.DataFrame([features])

        # Predict using the loaded pipeline
        preprocessor = self.pipeline['preprocessor']
        models = self.pipeline['models']
        features_processed = preprocessor.transform(features_df)
        
        price_prediction = {}
        for name, model in models.items():
            price_prediction[name] = round(model.predict(features_processed)[0], 2)
            
        return {
            "model_appraised": model_name,
            "grade": grade,
            "predicted_price_range": price_prediction
        }

if __name__ == "__main__":
    print("--- Price Prediction Tool ---")
    predictor = PricePredictor()
    
    result = predictor.predict(
        model_name='iPhone 14 Pro Max 128GB',
        grade='A',
        appraisal_date_str='2025-07-15'
    )
    
    print("\nAppraisal Result:")
    print(result)
