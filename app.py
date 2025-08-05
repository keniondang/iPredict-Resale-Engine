# -*- coding: utf-8 -*-
"""
A self-contained Flask API server for the used phone business.

This application loads all pre-trained models and data artifacts at startup
and exposes them through a series of API endpoints. It follows a unified
format, containing all necessary prediction and data-access logic internally.
"""

# 1. IMPORTS
# ==============================================================================
import pandas as pd
import numpy as np
import joblib
import re
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for Matplotlib

# 2. PREDICTION & DATA-ACCESS CLASSES
# ==============================================================================

class PricePredictor:
    """Loads and uses the pre-trained price prediction model."""
    def __init__(self, model_path='models/price_model.joblib', products_path='data/products.csv'):
        self.pipeline = joblib.load(model_path)
        self.products_df = pd.read_csv(products_path)
        self.products_df = self.products_df[self.products_df['product_type'] == 'Used Phone'].copy()
        self.products_df['base_model'] = self.products_df['model_name'].str.extract(r'(iPhone \d{1,2}(?: Pro| Plus| Pro Max| Mini)?)')[0]
        for col in ['release_date', 'successor_release_date']:
            self.products_df[col] = pd.to_datetime(self.products_df[col], errors='coerce')

    def predict(self, model_name: str, grade: str, appraisal_date_str: str):
        appraisal_date = pd.to_datetime(appraisal_date_str)
        phone_details_df = self.products_df[self.products_df['model_name'] == model_name]
        if phone_details_df.empty: return {"error": f"Model '{model_name}' not found."}
        phone_details = phone_details_df.iloc[0]
        features = {'base_model': phone_details['base_model'], 'grade': grade, 'storage_gb': phone_details['storage_gb'], 'model_tier': phone_details['model_tier'], 'original_msrp': phone_details['original_msrp'], 'days_since_model_release': (appraisal_date - phone_details['release_date']).days, 'days_since_successor_release': (appraisal_date - phone_details['successor_release_date']).days if pd.notna(phone_details['successor_release_date']) else -999}
        features_df = pd.DataFrame([features])
        features_processed = self.pipeline['preprocessor'].transform(features_df)
        models = self.pipeline['models']
        price_prediction = {name: round(model.predict(features_processed)[0], 2) for name, model in models.items()}
        return {"model_appraised": model_name, "grade": grade, "predicted_price_range": price_prediction}

class VelocityPredictor:
    """Loads and uses the pre-trained sales velocity model."""
    def __init__(self, model_path='models/velocity_model.joblib', products_path='data/products.csv', inventory_path='data/inventory_units.csv', transactions_path='data/transactions.csv'):
        self.pipeline = joblib.load(model_path)
        self.products_df = pd.read_csv(products_path)
        inventory = pd.read_csv(inventory_path)
        transactions = pd.read_csv(transactions_path)
        self.products_df = self.products_df[self.products_df['product_type'] == 'Used Phone'].copy()
        self.products_df['base_model'] = self.products_df['model_name'].str.extract(r'(iPhone \d{1,2}(?: Pro| Plus| Pro Max| Mini)?)')[0]
        self.products_df['release_date'] = pd.to_datetime(self.products_df['release_date'], errors='coerce')
        sold_units = pd.merge(inventory[inventory['status'] == 'Sold'], transactions[['unit_id', 'transaction_date']], on='unit_id')
        sold_units['transaction_date'] = pd.to_datetime(sold_units['transaction_date'])
        self.sales_history = pd.merge(sold_units, self.products_df[['product_id', 'base_model']], on='product_id')

    def _get_live_market_demand(self, base_model: str, appraisal_date: datetime):
        demand = self.sales_history[(self.sales_history['base_model'] == base_model) & (self.sales_history['transaction_date'] < appraisal_date) & (self.sales_history['transaction_date'] >= appraisal_date - timedelta(days=14))].shape[0]
        if demand == 0:
            match = re.search(r'(\d+)', base_model)
            if match:
                predecessor = base_model.replace(str(match.group(1)), str(int(match.group(1)) - 1))
                if predecessor in self.products_df['base_model'].unique():
                    pred_sales = self.sales_history[(self.sales_history['base_model'] == predecessor) & (self.sales_history['transaction_date'] < appraisal_date) & (self.sales_history['transaction_date'] >= appraisal_date - timedelta(days=90))]
                    if not pred_sales.empty: demand = int(np.ceil(pred_sales.shape[0] / (90 / 7.0)))
        return demand

    def predict(self, model_name: str, grade: str, appraisal_date_str: str):
        appraisal_date = pd.to_datetime(appraisal_date_str)
        phone_details_df = self.products_df[self.products_df['model_name'] == model_name]
        if phone_details_df.empty: return {"error": f"Model '{model_name}' not found."}
        phone_details = phone_details_df.iloc[0]
        features = {'grade': grade, 'storage_gb': phone_details['storage_gb'], 'model_tier': phone_details['model_tier'], 'original_msrp': phone_details['original_msrp'], 'days_since_model_release': (appraisal_date - phone_details['release_date']).days, 'market_demand': self._get_live_market_demand(phone_details['base_model'], appraisal_date)}
        prediction = self.pipeline.predict(pd.DataFrame([features]))[0]
        return {"model_appraised": model_name, "grade": grade, "predicted_sales_velocity": prediction}

class DynamicPricePredictor:
    """Loads and uses the pre-trained dynamic selling price model."""
    def __init__(self, model_path='models/dynamic_selling_price_pipeline.joblib', products_path='data/products.csv', inventory_path='data/inventory_units.csv', transactions_path='data/transactions.csv'):
        self.pipeline = joblib.load(model_path)
        self.products_df = pd.read_csv(products_path)
        self.inventory_df = pd.read_csv(inventory_path)
        transactions = pd.read_csv(transactions_path)
        self._prepare_reference_data(transactions)

    def _prepare_reference_data(self, transactions):
        self.products_df = self.products_df[self.products_df['product_type'] == 'Used Phone'].copy()
        self.products_df['base_model'] = self.products_df['model_name'].str.extract(r'(iPhone \d{1,2}(?: Pro| Plus| Pro Max| Mini)?)')[0]
        for col in ['release_date', 'successor_release_date']: self.products_df[col] = pd.to_datetime(self.products_df[col], errors='coerce')
        self.inventory_df['acquisition_date'] = pd.to_datetime(self.inventory_df['acquisition_date'])
        transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
        sold_inventory_info = pd.merge(self.inventory_df, self.products_df[['product_id', 'base_model']], on='product_id')
        self.sales_history = pd.merge(transactions, sold_inventory_info[['unit_id', 'base_model']], on='unit_id')

    def _get_live_market_demand(self, base_model: str, prediction_date: datetime):
        def get_predecessor(bm):
            match = re.search(r'(\d+)', bm)
            return bm.replace(str(match.group(1)), str(int(match.group(1)) - 1)) if match else None
        demand = self.sales_history[(self.sales_history['base_model'] == base_model) & (self.sales_history['transaction_date'] < prediction_date) & (self.sales_history['transaction_date'] >= prediction_date - timedelta(days=7))].shape[0]
        if demand == 0:
            predecessor = get_predecessor(base_model)
            if predecessor and predecessor in self.products_df['base_model'].unique():
                demand = self.sales_history[(self.sales_history['base_model'] == predecessor) & (self.sales_history['transaction_date'] < prediction_date) & (self.sales_history['transaction_date'] >= prediction_date - timedelta(days=7))].shape[0]
        return demand

    def predict(self, unit_id: int, prediction_date_str: str):
        prediction_date = pd.to_datetime(prediction_date_str)
        unit_info = self.inventory_df[self.inventory_df['unit_id'] == unit_id]
        if unit_info.empty: return {"error": f"Unit ID {unit_id} not found."}
        full_details = pd.merge(unit_info, self.products_df, on='product_id').iloc[0]
        features = {'original_msrp': full_details['original_msrp'], 'storage_gb': full_details['storage_gb'], 'grade': full_details['grade'], 'model_tier': full_details['model_tier'], 'base_model': full_details['base_model'], 'acquisition_price': full_details['acquisition_price'], 'days_in_inventory': (prediction_date - full_details['acquisition_date']).days, 'days_since_model_release': (prediction_date - full_details['release_date']).days, 'days_since_successor_release': (prediction_date - full_details['successor_release_date']).days if pd.notna(full_details['successor_release_date']) else -999, 'market_demand_7_days': self._get_live_market_demand(full_details['base_model'], prediction_date)}
        features_processed = self.pipeline['preprocessor'].transform(pd.DataFrame([features]))
        models = self.pipeline['models']
        price_prediction = {name: round(model.predict(features_processed)[0], 2) for name, model in models.items()}
        return {"unit_id": unit_id, "model_name": full_details['model_name'], "status": full_details['status'], "days_in_inventory": features['days_in_inventory'], "predicted_selling_price_range": price_prediction}

class Recommender:
    """Loads pre-built recommendation data to provide accessory suggestions."""
    def __init__(self, data_path='models/recommendation_data.joblib', products_path='data/products.csv'):
        recommendation_data = joblib.load(data_path)
        self.products_df = pd.read_csv(products_path)
        self.accessory_popularity = recommendation_data['popularity']
        self.compatibility_df = recommendation_data['compatibility']
        self.co_purchase_df = recommendation_data['co_purchase']
        self.phones_df = self.products_df[self.products_df['product_type'] == 'Used Phone'].copy()
        self.phones_df['base_model'] = self.phones_df['model_name'].str.extract(r'(iPhone \d{1,2}(?: Pro| Plus| Pro Max| Mini)?)')[0]

    def recommend_accessories(self, phone_product_id: int, top_n: int = 3, pop_weight: float = 0.4, copurchase_weight: float = 0.6):
        phone_info = self.phones_df[self.phones_df['product_id'] == phone_product_id]
        if phone_info.empty: return []
        phone_base_model = phone_info['base_model'].iloc[0]
        compatible_ids = self.compatibility_df[self.compatibility_df['phone_product_id'] == phone_product_id]['accessory_product_id'].tolist()
        if not compatible_ids: return []
        scores = {}
        for acc_id in compatible_ids:
            pop_score = self.accessory_popularity.get(acc_id, 0)
            co_purchase_entry = self.co_purchase_df[(self.co_purchase_df['phone_base_model'] == phone_base_model) & (self.co_purchase_df['accessory_product_id'] == acc_id)]
            co_purchase_score = co_purchase_entry['count'].iloc[0] if not co_purchase_entry.empty else 0
            scores[acc_id] = {'pop': pop_score, 'co': co_purchase_score}
        if not scores: return []
        scores_df = pd.DataFrame.from_dict(scores, orient='index')
        scores_df['pop_norm'] = scores_df['pop'] / scores_df['pop'].max() if scores_df['pop'].sum() != 0 else 0
        scores_df['co_norm'] = scores_df['co'] / scores_df['co'].max() if scores_df['co'].sum() != 0 else 0
        scores_df['final_score'] = (scores_df['pop_norm'] * pop_weight) + (scores_df['co_norm'] * copurchase_weight)
        return list(scores_df.sort_values(by='final_score', ascending=False).head(top_n).index)


# 3. FLASK APP INITIALIZATION & ENDPOINTS
# ==============================================================================
app = Flask(__name__)
CORS(app)

# --- Load Predictors and Data at Startup ---
try:
    print("--- Initializing Predictor Classes and Data Engines ---")
    price_predictor = PricePredictor()
    dynamic_price_predictor = DynamicPricePredictor()
    velocity_predictor = VelocityPredictor()
    recommender = Recommender()
    products_df_global = pd.read_csv('data/products.csv')
    discontinuation_list_global = joblib.load('models/discontinuation_list.joblib')
    print("--- All predictors and data loaded successfully. API is ready. ---")
except Exception as e:
    print(f"FATAL ERROR: Failed to initialize predictors. Details: {e}")
    price_predictor = None
    dynamic_price_predictor = None
    velocity_predictor = None
    recommender = None
    products_df_global = None
    discontinuation_list_global = None

@app.route('/api/products', methods=['GET'])
def get_products():
    """Endpoint to get a list of all available phone models for frontend dropdowns."""
    if products_df_global is None:
        return jsonify({"error": "Products data not loaded"}), 500
    try:
        phone_models = products_df_global[products_df_global['product_type'] == 'Used Phone']['model_name'].unique().tolist()
        return jsonify(sorted(phone_models))
    except Exception as e:
        return jsonify({"error": f"Could not load products: {e}"}), 500

@app.route('/api/appraise/buy', methods=['POST'])
def appraise_buy():
    """Combined endpoint to predict buying price and sales velocity."""
    if not price_predictor or not velocity_predictor:
        return jsonify({"error": "Predictor models not loaded"}), 500
    data = request.get_json()
    model_name, grade = data.get('model_name'), data.get('grade')
    if not model_name or not grade:
        return jsonify({"error": "model_name and grade are required"}), 400
    date_str = datetime.now().strftime('%Y-%m-%d')
    try:
        price_result = price_predictor.predict(model_name, grade, date_str)
        velocity_result = velocity_predictor.predict(model_name, grade, date_str)
        if "error" in price_result: return jsonify(price_result), 404
        if "error" in velocity_result: return jsonify(velocity_result), 404
        response = {"model_appraised": price_result.get("model_appraised"), "grade": price_result.get("grade"), "predicted_price_range": price_result.get("predicted_price_range"), "predicted_sales_velocity": velocity_result.get("predicted_sales_velocity")}
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

@app.route('/api/inventory/price', methods=['POST'])
def predict_dynamic_sell_price():
    """Endpoint to predict the dynamic selling price for an inventory item."""
    if not dynamic_price_predictor:
        return jsonify({"error": "Dynamic price predictor not loaded"}), 500
    data = request.get_json()
    unit_id = data.get('unit_id')
    if not unit_id: return jsonify({"error": "unit_id is required"}), 400
    date_str = datetime.now().strftime('%Y-%m-%d')
    try:
        result = dynamic_price_predictor.predict(int(unit_id), date_str)
        if "error" in result: return jsonify(result), 404
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

@app.route('/api/recommend/accessories', methods=['POST'])
def get_accessory_recommendations():
    """Endpoint to recommend accessories for a given phone."""
    if not recommender or products_df_global is None:
        return jsonify({"error": "Recommender system not loaded"}), 500
    data = request.get_json()
    phone_product_id = data.get('phone_product_id')
    if not phone_product_id: return jsonify({"error": "phone_product_id is required"}), 400
    try:
        recommended_ids = recommender.recommend_accessories(phone_product_id=int(phone_product_id), top_n=3)
        if not recommended_ids: return jsonify([])
        recommended_products = products_df_global[products_df_global['product_id'].isin(recommended_ids)]
        recommended_products = recommended_products.set_index('product_id').loc[recommended_ids].reset_index()
        return jsonify(recommended_products.to_dict('records'))
    except Exception as e:
        return jsonify({"error": f"Could not generate recommendations: {e}"}), 500

@app.route('/api/inventory/discontinuation_alerts', methods=['GET'])
def get_discontinuation_list():
    """Endpoint that returns the pre-built list of accessories to discontinue."""
    if discontinuation_list_global is None:
        return jsonify({"error": "Discontinuation list not loaded"}), 500
    return jsonify(discontinuation_list_global)

@app.route('/api/demand_forecast', methods=['POST'])
def get_product_demand_forecast():
    """Loads a specific Prophet model and returns forecast data as JSON."""
    data = request.get_json()
    category_value, periods = data.get('product_name'), data.get('periods', 90)
    if not category_value: return jsonify({"error": "product_name is required"}), 400
    safe_category_value = "".join(c for c in category_value if c.isalnum())
    model_path = os.path.join('models', f'demand_forecast_{safe_category_value}.joblib')
    try:
        model_package = joblib.load(model_path)
        model, release_date, successor_date, base_cap = model_package['model'], model_package['release_date'], model_package['successor_date'], model_package['base_cap']
    except FileNotFoundError:
        return jsonify({"error": f"Forecast model for '{category_value}' not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Could not load forecast model: {e}"}), 500
    try:
        future_df = model.make_future_dataframe(periods=int(periods))
        def calculate_cap(row_date):
            days_since_release = (row_date - release_date).days
            decay_factor = 1 - (days_since_release / (365 * 4))
            if pd.notna(successor_date) and row_date > successor_date:
                days_after_successor = (row_date - successor_date).days
                successor_decay = 0.5 - (days_after_successor / (365 * 2))
                decay_factor = min(decay_factor, successor_decay)
            return max(1, base_cap * decay_factor)
        future_df['cap'] = future_df['ds'].apply(calculate_cap)
        future_df['floor'] = 0
        forecast_df = model.predict(future_df)
        response_data = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(int(periods)).copy()
        response_data.rename(columns={'ds': 'date', 'yhat': 'prediction', 'yhat_lower': 'lower_bound', 'yhat_upper': 'upper_bound'}, inplace=True)
        response_data['date'] = response_data['date'].dt.strftime('%Y-%m-%d')
        return jsonify(response_data.to_dict('records'))
    except Exception as e:
        return jsonify({"error": f"Failed to generate forecast: {e}"}), 500

# 4. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    # Runs the Flask app. `debug=True` allows for auto-reloading when you save changes.
    # In a production environment, you would use a proper WSGI server like Gunicorn.
    app.run(debug=True, port=5000)
