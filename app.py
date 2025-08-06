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

class VelocityPredictor:
    """Uses the pre-trained sales velocity model."""
    def __init__(self, model_path, products_df, inventory_df, transactions_df):
        self.pipeline = joblib.load(model_path)
        products_df_phones = products_df[products_df['product_type'] == 'Used Phone'].copy()
        products_df_phones['base_model'] = products_df_phones['model_name'].str.extract(r'(iPhone \d{1,2}(?: Pro| Plus| Pro Max| Mini)?)')[0]
        products_df_phones['release_date'] = pd.to_datetime(products_df_phones['release_date'], errors='coerce')
        self.products_df = products_df_phones
        sold_units = pd.merge(inventory_df[inventory_df['status'] == 'Sold'], transactions_df[['unit_id', 'transaction_date']], on='unit_id')
        sold_units['transaction_date'] = pd.to_datetime(sold_units['transaction_date'])
        self.sales_history = pd.merge(sold_units, self.products_df[['product_id', 'base_model']], on='product_id')

    def _get_live_market_demand(self, base_model: str, appraisal_date: datetime):
        demand = self.sales_history[(self.sales_history['base_model'] == base_model) & (self.sales_history['transaction_date'] < appraisal_date) & (self.sales_history['transaction_date'] >= appraisal_date - timedelta(days=30))].shape[0]
        if demand == 0:
            match = re.search(r'(\d+)', base_model)
            if match:
                predecessor = base_model.replace(str(match.group(1)), str(int(match.group(1)) - 1))
                if predecessor in self.products_df['base_model'].unique():
                    pred_sales = self.sales_history[(self.sales_history['base_model'] == predecessor) & (self.sales_history['transaction_date'] < appraisal_date) & (self.sales_history['transaction_date'] >= appraisal_date - timedelta(days=90))]
                    if not pred_sales.empty: demand = int(np.ceil(pred_sales.shape[0] / (90 / 7.0)))
        return demand

    # --- MODIFICATION START ---
    # The method now returns the features used for generating an explanation.
    def predict(self, model_name: str, grade: str, appraisal_date_str: str):
        appraisal_date = pd.to_datetime(appraisal_date_str)
        phone_details_df = self.products_df[self.products_df['model_name'] == model_name]
        if phone_details_df.empty: return {"error": f"Model '{model_name}' not found."}
        phone_details = phone_details_df.iloc[0]
        days_since_release = (appraisal_date - phone_details['release_date']).days
        market_demand = self._get_live_market_demand(phone_details['base_model'], appraisal_date)
        
        features = {
            'grade': grade, 
            'storage_gb': phone_details['storage_gb'], 
            'model_tier': phone_details['model_tier'], 
            'original_msrp': phone_details['original_msrp'], 
            'days_since_model_release': days_since_release, 
            'market_demand': market_demand
        }
        
        prediction = self.pipeline.predict(pd.DataFrame([features]))[0]
        
        return {
            "model_appraised": model_name, 
            "grade": grade, 
            "predicted_sales_velocity": prediction,
            "market_demand": market_demand,
            "days_since_release": days_since_release
        }
    # --- MODIFICATION END ---

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
    velocity_predictor = VelocityPredictor('models/velocity_model.joblib', products_df_global, inventory_df_global, transactions_df_global)
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

# --- MODIFICATION START ---
# Helper function to generate a human-readable reason for the velocity prediction.
def _generate_velocity_reason(velocity_result: dict) -> str:
    """Generates a human-readable explanation for the sales velocity prediction."""
    velocity = velocity_result.get("predicted_sales_velocity")
    demand = velocity_result.get("market_demand", 0)
    days_since_release = velocity_result.get("days_since_release", 0)

    # Convert days to a more readable format (years/months)
    if days_since_release > 365:
        age_str = f"{days_since_release / 365:.1f} years"
    else:
        age_str = f"{days_since_release} days"

    # Define reason templates based on the new, less aggressive thresholds
    reasons = {
        "Fast Mover": f"Predicted to sell in under 45 days. This model is relatively recent or has strong market demand ({demand} units sold in the last 30 days).",
        "Medium Mover": f"Predicted to sell in 45-120 days. Market demand is moderate ({demand} units sold in the last 30 days).",
        "Dead Stock Risk": f"Predicted to take over 120 days to sell. This is likely due to low market demand ({demand} units sold in last 30 days) and the model's age ({age_str} since release)."
    }
    return reasons.get(velocity, "Reason could not be determined.")
# --- MODIFICATION END ---

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

        # --- MODIFICATION START ---
        # Get velocity prediction and generate the reason string
        velocity_result = velocity_predictor.predict(model_name, grade, date_str)
        if "error" in velocity_result: return jsonify(velocity_result), 404
        velocity_reason = _generate_velocity_reason(velocity_result)
        # --- MODIFICATION END ---

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

        # --- MODIFICATION START ---
        # Add the velocity reason to the response object
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
        # --- MODIFICATION END ---
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

# Add this endpoint in your app.py file
@app.route('/api/filter_options', methods=['GET'])
def get_filter_options():
    """Endpoint to get unique store names and phone models for filtering."""
    if not db_engine:
        return jsonify({"error": "Database connection not available"}), 500
    try:
        with db_engine.connect() as connection:
            stores_query = text("SELECT DISTINCT store_name FROM stores ORDER BY store_name;")
            stores = [row[0] for row in connection.execute(stores_query).fetchall()]

            models_query = text("""
                SELECT DISTINCT p.model_name
                FROM products p
                INNER JOIN inventory_units i ON p.product_id = i.product_id
                WHERE i.status = 'Sold'
                ORDER BY p.model_name;
            """)
            models = [row[0] for row in connection.execute(models_query).fetchall()]

        return jsonify({
            "stores": stores,
            "models": models
        })
    except Exception as e:
        print(f"ERROR in get_filter_options: {e}")
        return jsonify({"error": f"Could not fetch filter options: {str(e)}"}), 500


# Replace your existing get_transaction_history function with this one
@app.route('/api/transactions/history', methods=['GET'])
def get_transaction_history():
    """Endpoint to get a summary and detailed list of all sales transactions, with filtering."""
    if not db_engine:
        return jsonify({"error": "Database connection not available"}), 500

    # Get filter parameters from the request URL
    store_filter = request.args.get('store', None)
    model_filter = request.args.get('model', None)
    start_date_filter = request.args.get('start_date', None)
    end_date_filter = request.args.get('end_date', None)

    base_query = """
        SELECT
            t.transaction_id, t.transaction_date,
            p.model_name, i.grade, s.store_name,
            i.acquisition_price, t.final_sale_price
        FROM transactions t
        INNER JOIN inventory_units i ON t.unit_id = i.unit_id
        INNER JOIN products p ON i.product_id = p.product_id
        INNER JOIN stores s ON i.store_id = s.store_id
    """
    conditions = ["i.status = 'Sold'"]
    params = {}

    if store_filter:
        conditions.append("s.store_name = :store")
        params['store'] = store_filter
    if model_filter:
        conditions.append("p.model_name = :model")
        params['model'] = model_filter
    if start_date_filter:
        conditions.append("t.transaction_date >= :start_date")
        params['start_date'] = start_date_filter
    if end_date_filter:
        conditions.append("t.transaction_date <= :end_date")
        params['end_date'] = end_date_filter

    query = base_query + " WHERE " + " AND ".join(conditions) + " ORDER BY t.transaction_date DESC;"

    try:
        with db_engine.connect() as connection:
            result = connection.execute(text(query), params)
            rows = result.fetchall()
            columns = result.keys()

        if not rows:
            return jsonify({
                "summary": {"total_revenue": 0, "total_cost": 0, "total_profit": 0, "units_sold": 0},
                "history": []
            })

        history_df = pd.DataFrame(rows, columns=columns)
        history_df['acquisition_price'] = pd.to_numeric(history_df['acquisition_price'], errors='coerce').fillna(0)
        history_df['final_sale_price'] = pd.to_numeric(history_df['final_sale_price'], errors='coerce').fillna(0)
        history_df['profit'] = history_df['final_sale_price'] - history_df['acquisition_price']

        summary = {
            "total_revenue": round(float(history_df['final_sale_price'].sum()), 2),
            "total_cost": round(float(history_df['acquisition_price'].sum()), 2),
            "total_profit": round(float(history_df['profit'].sum()), 2),
            "units_sold": int(len(history_df))
        }

        history_df['transaction_date'] = pd.to_datetime(history_df['transaction_date']).dt.strftime('%Y-%m-%d')
        history_df = history_df.replace({np.nan: None})
        history_list = history_df.to_dict('records')

        return jsonify({"summary": summary, "history": history_list})

    except Exception as e:
        print(f"--- DETAILED DATABASE ERROR ---")
        print(f"ERROR TYPE: {type(e).__name__}")
        print(f"ERROR DETAILS: {e}")
        print(f"-------------------------------")
        return jsonify({"error": f"A database error occurred. Please check the server console for detailed logs."}), 500

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