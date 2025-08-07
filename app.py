# -*- coding: utf-8 -*-
"""
A self-contained Flask API server for the used phone business.
This application loads all pre-trained models at startup and connects
to a SQL Server database for all live data access.
"""

# 1. IMPORTS
# ==============================================================================
import pandas as pd
import numpy as np
import joblib
import re
from sqlalchemy import text
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import urllib
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for Matplotlib

# 2. DATABASE CONNECTION SETUP
# ==============================================================================

# --- CONFIGURATION ---
# !!! IMPORTANT: UPDATE THESE VALUES TO MATCH YOUR SQL SERVER SETUP !!!
SERVER_NAME = "localhost\\SQLEXPRESS"
DATABASE_NAME = "UsedPhoneResale"
# ---------------------

def get_db_engine():
    """Creates and returns a SQLAlchemy engine for SQL Server."""
    try:
        params = urllib.parse.quote_plus(
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={SERVER_NAME};"
            f"DATABASE={DATABASE_NAME};"
            f"Trusted_Connection=yes;"
        )
        connection_string = f"mssql+pyodbc:///?odbc_connect={params}"
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        print(f"FATAL: Could not create database engine. Error: {e}")
        return None

# 3. PREDICTION & DATA-ACCESS CLASSES
# ==============================================================================

class PricePredictor:
    """Uses the pre-trained price prediction model."""
    def __init__(self, model_path, products_df):
        self.pipeline = joblib.load(model_path)
        self.products_df = products_df[products_df['product_type'] == 'Used Phone'].copy()
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
        raw_predictions = [
            models['low'].predict(features_processed)[0],
            models['median'].predict(features_processed)[0],
            models['high'].predict(features_processed)[0]
        ]
        sorted_predictions = sorted(raw_predictions)
        price_prediction = {
            "low": round(sorted_predictions[0], 2),
            "median": round(sorted_predictions[1], 2),
            "high": round(sorted_predictions[2], 2)
        }
        return {"model_appraised": model_name, "grade": grade, "predicted_price_range": price_prediction}

# Delete the old VelocityPredictor class in app.py and replace it with this

class AdvancedVelocityPredictor:
    """Uses the pre-trained ADVANCED sales velocity model."""
    def __init__(self, model_path, products_df):
        try:
            pipeline = joblib.load(model_path)
            self.model = pipeline['model']
            self.features = pipeline['features']
            self.historical_data = pipeline['historical_data']
            self.products_df = products_df
            print("Successfully loaded ADVANCED velocity model pipeline.")
        except Exception as e:
            print(f"FATAL: Could not load advanced velocity model. Error: {e}")
            raise e

    def predict(self, model_name: str, grade: str):
        """
        Generates a sales velocity prediction by looking up the most recent
        historical data for a specific model and grade.
        """
        # Find the product_id for the given model_name
        product_info = self.products_df[self.products_df['model_name'] == model_name]
        if product_info.empty:
            return {"error": f"Model '{model_name}' not found."}
        
        product_id = product_info.iloc[0]['product_id']
        product_details = self.products_df[self.products_df['product_id'] == product_id].iloc[0]

        # Get the most recent feature values for this specific product and grade
        item_history = self.historical_data[
            (self.historical_data['product_id'] == product_id) & 
            (self.historical_data['grade'] == grade)
        ].sort_values('transaction_date', ascending=False)

        if item_history.empty:
            # If no history for this specific grade, assume 0 for features
            sales_count_recent = 0
            avg_days_recent = 0
            reason = f"No sales history found for a Grade '{grade}' {model_name}. Predicting based on model tier and MSRP only."
        else:
            # Use the latest available historical record for features
            latest_stats = item_history.iloc[0]
            sales_count_recent = latest_stats['grade_specific_sales_last_120d']
            avg_days_recent = latest_stats['grade_specific_avg_days_last_120d']
            reason = f"Based on historical data for Grade '{grade}' units: The average time to sell was {avg_days_recent:.1f} days, with {int(sales_count_recent)} units sold in the prior 120-day period."

        # Construct the feature DataFrame for prediction
        prediction_data = pd.DataFrame([{
            'model_tier': product_details['model_tier'],
            'storage_gb': product_details['storage_gb'],
            'original_msrp': product_details['original_msrp'],
            'grade_specific_sales_last_120d': sales_count_recent,
            'grade_specific_avg_days_last_120d': avg_days_recent
        }])
        
        # Ensure correct feature order
        prediction_data = prediction_data[self.features]

        predicted_category = self.model.predict(prediction_data)[0]

        return {
            "model_appraised": model_name,
            "grade": grade,
            "predicted_sales_velocity": predicted_category,
            "velocity_reason": reason # The new model provides a more direct reason
        }

class DynamicPricePredictor:
    """Uses the pre-trained dynamic selling price model."""
    def __init__(self, model_path, products_df, inventory_df, transactions_df):
        self.pipeline = joblib.load(model_path)
        self.inventory_df = inventory_df.copy()
        self.inventory_df['acquisition_date'] = pd.to_datetime(self.inventory_df['acquisition_date'])
        products_df_phones = products_df[products_df['product_type'] == 'Used Phone'].copy()
        products_df_phones['base_model'] = products_df_phones['model_name'].str.extract(r'(iPhone \d{1,2}(?: Pro| Plus| Pro Max| Mini)?)')[0]
        for col in ['release_date', 'successor_release_date']:
            products_df_phones[col] = pd.to_datetime(products_df_phones[col], errors='coerce')
        self.products_df = products_df_phones
        transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
        sold_inventory_info = pd.merge(self.inventory_df, self.products_df[['product_id', 'base_model']], on='product_id')
        self.sales_history = pd.merge(transactions_df, sold_inventory_info[['unit_id', 'base_model']], on='unit_id')

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
        features = {
            'original_msrp': full_details['original_msrp'], 'storage_gb': full_details['storage_gb'], 'grade': full_details['grade'], 'model_tier': full_details['model_tier'], 'base_model': full_details['base_model'], 'days_since_model_release': (prediction_date - full_details['release_date']).days, 'days_since_successor_release': (prediction_date - full_details['successor_release_date']).days if pd.notna(full_details['successor_release_date']) else -999, 'market_demand_7_days': self._get_live_market_demand(full_details['base_model'], prediction_date)
        }
        features_processed = self.pipeline['preprocessor'].transform(pd.DataFrame([features]))
        models = self.pipeline['models']
        price_prediction = {name: round(model.predict(features_processed)[0], 2) for name, model in models.items()}
        return {
            "unit_id": unit_id, "model_name": full_details['model_name'], "status": full_details['status'], "days_in_inventory": (prediction_date - full_details['acquisition_date']).days, "acquisition_date": full_details['acquisition_date'].strftime('%Y-%m-%d'), "acquisition_price": full_details['acquisition_price'], "predicted_selling_price_range": price_prediction
        }

class Recommender:
    """Uses pre-built recommendation data to provide accessory suggestions."""
    def __init__(self, data_path, products_df):
        recommendation_data = joblib.load(data_path)
        self.products_df = products_df
        self.accessory_popularity = recommendation_data['popularity']
        self.compatibility_df = recommendation_data['compatibility']
        self.co_purchase_df = recommendation_data['co_purchase']
        self.phones_df = self.products_df[self.products_df['product_type'] == 'Used Phone'].copy()
        self.phones_df['base_model'] = self.phones_df['model_name'].str.extract(r'(iPhone \d{1,2}(?: Pro| Plus| Pro Max| Mini)?)')[0]

    def recommend_accessories(self, phone_model_name: str, top_n: int = 3, pop_weight: float = 0.4, copurchase_weight: float = 0.6):
        phone_info = self.phones_df[self.phones_df['model_name'] == phone_model_name]
        if phone_info.empty: return []

        phone_product_id = phone_info['product_id'].iloc[0]
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


# 4. FLASK APP INITIALIZATION & ENDPOINTS
# ==============================================================================
app = Flask(__name__)
CORS(app)
db_engine = get_db_engine()

# --- Load Predictors and Data at Startup ---
try:
    print("--- Initializing Predictor Classes and Loading Data from DB ---")
    products_df_global = pd.read_sql("SELECT * FROM products", db_engine)
    inventory_df_global = pd.read_sql("SELECT * FROM inventory_units", db_engine)
    transactions_df_global = pd.read_sql("SELECT * FROM transactions", db_engine)
    stores_df_global = pd.read_sql("SELECT * FROM stores", db_engine)
    price_predictor = PricePredictor('models/price_model.joblib', products_df_global)
    dynamic_price_predictor = DynamicPricePredictor('models/dynamic_selling_price_pipeline.joblib', products_df_global, inventory_df_global, transactions_df_global)
    velocity_predictor = AdvancedVelocityPredictor('models/velocity_model_advanced.joblib', products_df_global)
    recommender = Recommender('models/recommendation_data.joblib', products_df_global)
    discontinuation_list_global = joblib.load('models/discontinuation_list.joblib')
    print("--- All predictors and data loaded successfully. API is ready. ---")
except Exception as e:
    print(f"FATAL ERROR: Failed to initialize predictors or load data. Details: {e}")
    price_predictor, dynamic_price_predictor, velocity_predictor, recommender = None, None, None, None
    products_df_global, inventory_df_global, stores_df_global, discontinuation_list_global = None, None, None, None

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
    if not all([price_predictor, velocity_predictor, dynamic_price_predictor]):
        return jsonify({"error": "Predictor models not loaded"}), 500

    data = request.get_json()
    model_name = data.get('model_name')
    grade = data.get('grade')
    desired_profit = data.get('desired_profit', 0)

    if not model_name or not grade:
        return jsonify({"error": "model_name and grade are required"}), 400

    date_str = datetime.now().strftime('%Y-%m-%d')
    appraisal_date = pd.to_datetime(date_str)

    try:
        historical_buy_result = price_predictor.predict(model_name, grade, date_str)
        if "error" in historical_buy_result: return jsonify(historical_buy_result), 404

        velocity_result = velocity_predictor.predict(model_name, grade)
        if "error" in velocity_result: return jsonify(velocity_result), 404
        velocity_reason = velocity_result.get("velocity_reason") # Get the reason directly from the new predictor

        phone_details_df = dynamic_price_predictor.products_df[dynamic_price_predictor.products_df['model_name'] == model_name]
        if phone_details_df.empty: return jsonify({"error": f"Model '{model_name}' not found for selling price."}), 404

        full_details = phone_details_df.iloc[0]
        features = {
            'original_msrp': full_details['original_msrp'], 'storage_gb': full_details['storage_gb'], 'grade': grade,
            'model_tier': full_details['model_tier'], 'base_model': full_details['base_model'],
            'days_since_model_release': (appraisal_date - full_details['release_date']).days,
            'days_since_successor_release': (appraisal_date - full_details['successor_release_date']).days if pd.notna(full_details['successor_release_date']) else -999,
            'market_demand_7_days': dynamic_price_predictor._get_live_market_demand(full_details['base_model'], appraisal_date)
        }
        features_processed = dynamic_price_predictor.pipeline['preprocessor'].transform(pd.DataFrame([features]))
        d_models = dynamic_price_predictor.pipeline['models']
        selling_price_prediction = {name: round(model.predict(features_processed)[0], 2) for name, model in d_models.items()}

        profit_target_range = {
            "low": selling_price_prediction.get('low_liquidate_price', 0) - float(desired_profit),
            "median": selling_price_prediction.get('median_fair_market_price', 0) - float(desired_profit),
            "high": selling_price_prediction.get('high_start_price', 0) - float(desired_profit),
        }

        market_alert = False
        historical_median_buy = historical_buy_result.get("predicted_price_range", {}).get("median", 0)
        if historical_median_buy > profit_target_range["high"]:
            market_alert = True

        response = {
            "model_appraised": historical_buy_result.get("model_appraised"),
            "grade": historical_buy_result.get("grade"),
            "historically_based_buy_price": historical_buy_result.get("predicted_price_range"),
            "predicted_sales_velocity": velocity_result.get("predicted_sales_velocity"),
            "velocity_reason": velocity_reason,
            "predicted_selling_price_range": selling_price_prediction,
            "profit_targeted_buy_price_range": {k: round(v, 2) for k, v in profit_target_range.items()},
            "market_alert": market_alert
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

@app.route('/api/inventory/all', methods=['GET'])
def get_all_inventory():
    if inventory_df_global is None or products_df_global is None or stores_df_global is None:
        return jsonify({"error": "Inventory, products, or stores data not loaded"}), 500
    try:
        in_stock = inventory_df_global[inventory_df_global['status'] == 'In Stock'].copy()
        in_stock_details = pd.merge(in_stock, products_df_global[['product_id', 'model_name']], on='product_id', how='left')
        in_stock_with_stores = pd.merge(in_stock_details, stores_df_global[['store_id', 'store_name']], on='store_id', how='left')
        in_stock_with_stores['acquisition_date'] = pd.to_datetime(in_stock_with_stores['acquisition_date']).dt.strftime('%Y-%m-%d')
        grouped_inventory = in_stock_with_stores.groupby('store_name')
        response_data = {}
        for store_name, group_df in grouped_inventory:
            items = group_df[['unit_id', 'model_name', 'grade', 'acquisition_date']].to_dict('records')
            response_data[store_name] = items
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": f"Could not load inventory: {e}"}), 500

@app.route('/api/inventory/price', methods=['POST'])
def predict_dynamic_sell_price():
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
    if not recommender or products_df_global is None:
        return jsonify({"error": "Recommender system not loaded"}), 500
    data = request.get_json()
    phone_model_name = data.get('phone_model_name')
    if not phone_model_name: return jsonify({"error": "phone_model_name is required"}), 400
    try:
        recommended_ids = recommender.recommend_accessories(phone_model_name=phone_model_name, top_n=3)
        if not recommended_ids: return jsonify([])
        recommended_products = products_df_global[products_df_global['product_id'].isin(recommended_ids)]
        recommended_products = recommended_products.set_index('product_id').loc[recommended_ids].reset_index()
        recommended_products = recommended_products.replace({np.nan: None})
        return jsonify(recommended_products.to_dict('records'))
    except Exception as e:
        return jsonify({"error": f"Could not generate recommendations: {e}"}), 500

@app.route('/api/inventory/discontinuation_alerts', methods=['GET'])
def get_discontinuation_list():
    if discontinuation_list_global is None:
        return jsonify({"error": "Discontinuation list not loaded"}), 500
    return jsonify(discontinuation_list_global)

@app.route('/api/demand_forecast', methods=['POST'])
def get_product_demand_forecast():
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
            decay_factor = 1 - (days_since_release / (365 * 2))
            if pd.notna(successor_date) and row_date > successor_date:
                days_after_successor = (row_date - successor_date).days
                successor_decay = 0.5 - (days_after_successor / (365 * 1))
                decay_factor = min(decay_factor, successor_decay)
            return max(1, base_cap * decay_factor)
        future_df['cap'] = future_df['ds'].apply(calculate_cap)
        future_df['floor'] = 0
        forecast_df = model.predict(future_df)

        # Calculate total forecast for the period
        total_forecast = forecast_df['yhat'][-int(periods):].sum()

        # Prepare daily data for the chart
        daily_data = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(int(periods)).copy()
        daily_data.rename(columns={'ds': 'date', 'yhat': 'prediction', 'yhat_lower': 'lower_bound', 'yhat_upper': 'upper_bound'}, inplace=True)
        daily_data['date'] = daily_data['date'].dt.strftime('%Y-%m-%d')

        # Build the final response object
        response_data = {
            "daily_data": daily_data.to_dict('records'),
            "total_forecast": int(round(total_forecast))
        }
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": f"Failed to generate forecast: {e}"}), 500

# 5. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    app.run(debug=True, port=5000)