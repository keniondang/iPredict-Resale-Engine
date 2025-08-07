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
from sqlalchemy import text, func
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
from sqlalchemy import create_engine, inspect
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
        except Exception as e:
            print(f"FATAL: Could not load velocity model. Error: {e}")
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
    products_df_global['release_date'] = pd.to_datetime(products_df_global['release_date'])
    inventory_df_global = pd.read_sql("SELECT * FROM inventory_units", db_engine)
    transactions_df_global = pd.read_sql("SELECT * FROM transactions", db_engine)
    stores_df_global = pd.read_sql("SELECT * FROM stores", db_engine)
    price_predictor = PricePredictor('models/price_model.joblib', products_df_global)
    dynamic_price_predictor = DynamicPricePredictor('models/dynamic_selling_price_pipeline.joblib', products_df_global, inventory_df_global, transactions_df_global)
    velocity_predictor = AdvancedVelocityPredictor('models/velocity_model_advanced.joblib', products_df_global)
    recommender = Recommender('models/recommendation_data.joblib', products_df_global)
    discontinuation_list_global = joblib.load('models/discontinuation_list.joblib')
    # Create a set of product IDs for fast lookups
    discontinuation_id_set = {item['product_id'] for item in discontinuation_list_global}
    print("--- All predictors and data loaded successfully. API is ready. ---")
except Exception as e:
    print(f"FATAL ERROR: Failed to initialize predictors or load data. Details: {e}")
    price_predictor, dynamic_price_predictor, velocity_predictor, recommender = None, None, None, None
    products_df_global, inventory_df_global, stores_df_global, discontinuation_list_global = None, None, None, None
    discontinuation_id_set = set()

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

@app.route('/api/stores', methods=['GET'])
def get_stores():
    """Endpoint to get a list of all stores for filtering."""
    if stores_df_global is None:
        return jsonify({"error": "Stores data not loaded"}), 500
    try:
        # Return all stores except the 'Online Store' for purchasing context
        physical_stores = stores_df_global[stores_df_global['store_name'] != 'Online Store']
        stores_list = physical_stores[['store_id', 'store_name']].to_dict('records')
        return jsonify(sorted(stores_list, key=lambda x: x['store_name']))
    except Exception as e:
        return jsonify({"error": f"Could not load stores: {e}"}), 500


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

@app.route('/api/inventory/buy', methods=['POST'])
def buy_inventory():
    """Adds a new phone unit to the inventory_units table."""
    data = request.get_json()
    model_name = data.get('model_name')
    grade = data.get('grade')
    store_id = data.get('store_id')
    acquisition_price = data.get('acquisition_price')

    if not all([model_name, grade, store_id, acquisition_price]):
        return jsonify({"error": "model_name, grade, store_id, and acquisition_price are required"}), 400

    try:
        # Find product details from the global DataFrame
        product_details = products_df_global[products_df_global['model_name'] == model_name].iloc[0]
        product_id = product_details['product_id']
        release_date = product_details['release_date']
        
        acquisition_date = datetime.now()
        age_at_acquisition = (acquisition_date - release_date).days
        
        with db_engine.connect() as connection:
            transaction = connection.begin()
            try:
                # Get the current max unit_id to determine the new one safely
                max_unit_id_query = text("SELECT MAX(unit_id) FROM inventory_units")
                max_unit_id_result = connection.execute(max_unit_id_query).scalar()
                new_unit_id = (max_unit_id_result or 0) + 1

                insert_query = text("""
                    INSERT INTO inventory_units 
                    (unit_id, product_id, store_id, acquisition_date, grade, age_at_acquisition_days, acquisition_price, status)
                    VALUES 
                    (:unit_id, :product_id, :store_id, :acquisition_date, :grade, :age_at_acquisition_days, :acquisition_price, 'In Stock')
                """)
                
                connection.execute(insert_query, {
                    "unit_id": new_unit_id,
                    "product_id": int(product_id),
                    "store_id": int(store_id),
                    "acquisition_date": acquisition_date.strftime('%Y-%m-%d'),
                    "grade": grade,
                    "age_at_acquisition_days": age_at_acquisition,
                    "acquisition_price": float(acquisition_price)
                })
                
                transaction.commit()
                
                # Update the global inventory DataFrame to reflect the change immediately
                global inventory_df_global
                new_record = pd.DataFrame([{
                    "unit_id": new_unit_id,
                    "product_id": int(product_id),
                    "store_id": int(store_id),
                    "acquisition_date": acquisition_date,
                    "grade": grade,
                    "age_at_acquisition_days": age_at_acquisition,
                    "acquisition_price": float(acquisition_price),
                    "status": "In Stock"
                }])
                inventory_df_global = pd.concat([inventory_df_global, new_record], ignore_index=True)

                return jsonify({"success": True, "unit_id": new_unit_id})

            except Exception as e:
                transaction.rollback()
                print(f"Database transaction failed: {e}")
                return jsonify({"error": f"Database error: {e}"}), 500

    except IndexError:
        return jsonify({"error": f"Model '{model_name}' not found."}), 404
    except Exception as e:
        print(f"An error occurred in /api/inventory/buy: {e}")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


@app.route('/api/inventory/combined', methods=['GET'])
@app.route('/api/inventory/combined', methods=['GET'])
def get_combined_inventory():
    """ Endpoint to fetch and enrich both phone and accessory inventory. """
    try:
        # UPDATED: Added i.store_id to the phone query
        phone_query = text("""
            SELECT i.unit_id, i.grade, i.acquisition_date, p.model_name, s.store_name, i.store_id
            FROM inventory_units i
            JOIN products p ON i.product_id = p.product_id
            JOIN stores s ON i.store_id = s.store_id
            WHERE i.status = 'In Stock'
        """)
        # UPDATED: Added ai.store_id to the accessory query
        accessory_query = text("""
            SELECT ai.product_id, ai.quantity, p.model_name, s.store_name, ai.store_id
            FROM accessory_inventory ai
            JOIN products p ON ai.product_id = p.product_id
            JOIN stores s ON ai.store_id = s.store_id
            WHERE ai.quantity > 0
        """)
        
        with db_engine.connect() as connection:
            phones_df = pd.read_sql(phone_query, connection)
            accessories_df = pd.read_sql(accessory_query, connection)
            
        phones_df['acquisition_date'] = pd.to_datetime(phones_df['acquisition_date']).dt.strftime('%Y-%m-%d')
        
        stores_to_display = stores_df_global[stores_df_global['store_name'] != 'Online Store']
        
        response_data = {store_name: {
            "summary": {"phone_units": 0, "accessory_types": 0},
            "phones": [],
            "accessories": []
        } for store_name in stores_to_display['store_name'].unique()}

        if not phones_df.empty:
            phones_list = phones_df.to_dict('records')
            for phone in phones_list:
                prediction = velocity_predictor.predict(phone['model_name'], phone['grade'])
                phone['velocity_label'] = prediction.get('predicted_sales_velocity', 'Unknown')
            
            phones_df_enriched = pd.DataFrame(phones_list)
            grouped_phones = phones_df_enriched.groupby('store_name')
            for store_name, group_df in grouped_phones:
                if store_name in response_data:
                    response_data[store_name]['phones'] = group_df.to_dict('records')
                    response_data[store_name]['summary']['phone_units'] = len(group_df)

        if not accessories_df.empty:
            accessories_list = accessories_df.to_dict('records')
            for acc in accessories_list:
                if acc['product_id'] in discontinuation_id_set:
                    acc['discontinuation_label'] = 'Discontinuation Alert'
            
            accessories_df_enriched = pd.DataFrame(accessories_list)
            
            accessories_df_enriched.replace({np.nan: None}, inplace=True)
            
            grouped_accessories = accessories_df_enriched.groupby('store_name')
            for store_name, group_df in grouped_accessories:
                 if store_name in response_data:
                    response_data[store_name]['accessories'] = group_df.to_dict('records')
                    response_data[store_name]['summary']['accessory_types'] = len(group_df)
        
        return jsonify(response_data)

    except Exception as e:
        print(f"Error in /api/inventory/combined: {e}")
        return jsonify({"error": "Could not load combined inventory"}), 500
    

@app.route('/api/inventory/price', methods=['POST'])
def predict_dynamic_sell_price():
    if not dynamic_price_predictor:
        return jsonify({"error": "Dynamic price predictor not loaded"}), 500
    data = request.get_json()
    unit_id = data.get('unit_id')
    if not unit_id: return jsonify({"error": "unit_id is required"}), 400
    date_str = datetime.now().strftime('%Y-%m-%d')
    try:
        dynamic_price_predictor.inventory_df = inventory_df_global.copy()
        dynamic_price_predictor.inventory_df['acquisition_date'] = pd.to_datetime(dynamic_price_predictor.inventory_df['acquisition_date'])
        result = dynamic_price_predictor.predict(int(unit_id), date_str)
        if "error" in result: return jsonify(result), 404
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


@app.route('/api/accessories/details/<int:product_id>', methods=['GET'])
def get_accessory_details(product_id):
    """ Endpoint to get details and compatibility for a single accessory. """
    try:
        accessory_info_query = text("SELECT model_name, original_msrp FROM products WHERE product_id = :pid")
        compat_query = text("""
            SELECT p.model_name
            FROM accessory_compatibility ac
            JOIN products p ON ac.phone_product_id = p.product_id
            WHERE ac.accessory_product_id = :pid
            ORDER BY p.release_date DESC;
        """)

        with db_engine.connect() as connection:
            accessory_info = connection.execute(accessory_info_query, {"pid": product_id}).fetchone()
            if not accessory_info:
                return jsonify({"error": "Accessory not found"}), 404
            
            compat_list = connection.execute(compat_query, {"pid": product_id}).fetchall()
            compatible_phones = [row[0] for row in compat_list]

        response = {
            "product_id": product_id,
            "model_name": accessory_info[0],
            "original_msrp": accessory_info[1],
            "compatible_phones": compatible_phones
        }
        return jsonify(response)

    except Exception as e:
        print(f"Error getting accessory details for ID {product_id}: {e}")
        return jsonify({"error": "Could not retrieve accessory details"}), 500

@app.route('/api/transactions/history', methods=['GET'])
def get_transaction_history():
    """ Endpoint to retrieve and filter transaction history for both phones and accessories. """
    try:
        base_query = """
            -- Phone Sales
            SELECT
                t.transaction_id,
                t.transaction_date,
                p.model_name,
                p.product_type,
                i.grade,
                t.final_sale_price,
                i.acquisition_price AS cost,
                s.store_name,
                s.store_id,
                i.acquisition_date
            FROM transactions t
            JOIN inventory_units i ON t.unit_id = i.unit_id
            JOIN products p ON i.product_id = p.product_id
            JOIN stores s ON i.store_id = s.store_id
            WHERE p.product_type = 'Used Phone'

            UNION ALL

            -- Accessory Sales
            SELECT
                t.transaction_id,
                t.transaction_date,
                p.model_name,
                p.product_type,
                NULL AS grade,
                t.final_sale_price,
                p.original_msrp AS cost,
                s.store_name,
                s.store_id,
                NULL as acquisition_date
            FROM transactions t
            JOIN products p ON t.product_id = p.product_id
            JOIN stores s ON t.store_id = s.store_id
            WHERE p.product_type = 'Accessory'
        """
        
        with db_engine.connect() as connection:
            df = pd.read_sql_query(text(base_query), connection)

        # Sort by transaction_id descending
        df = df.sort_values('transaction_id', ascending=False)
        
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['acquisition_date'] = pd.to_datetime(df['acquisition_date'], errors='coerce')
        
        if request.args.get('model_name'):
            df = df[df['model_name'] == request.args.get('model_name')]
        if request.args.get('grade'):
            df = df[df['grade'] == request.args.get('grade')]
        if request.args.get('store_id'):
            df = df[df['store_id'] == int(request.args.get('store_id'))]
        if request.args.get('start_date'):
            df = df[df['transaction_date'] >= pd.to_datetime(request.args.get('start_date'))]
        if request.args.get('end_date'):
            df = df[df['transaction_date'] <= pd.to_datetime(request.args.get('end_date'))]

        if df.empty:
            return jsonify({"summary": {}, "transactions": []})

        df['profit_loss'] = df['final_sale_price'] - df['cost']
        df['days_in_stock'] = (df['transaction_date'] - df['acquisition_date']).dt.days

        phones_df = df[df['product_type'] == 'Used Phone'].copy()
        
        summary = {
            "total_revenue": df['final_sale_price'].sum(),
            "total_cost": df['cost'].sum(),
            "total_profit": df['profit_loss'].sum(),
            "total_items_sold": len(df),
            "average_profit_per_item": df['profit_loss'].mean(),
            "average_days_in_stock": phones_df['days_in_stock'].mean()
        }

        df['transaction_date'] = df['transaction_date'].dt.strftime('%Y-%m-%d')
        df.replace({np.nan: None, pd.NaT: None, np.inf: None, -np.inf: None}, inplace=True)
        transactions = df.to_dict('records')
        
        return jsonify({"summary": summary, "transactions": transactions})

    except Exception as e:
        print(f"Error in /api/transactions/history: {e}")
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

        total_forecast = forecast_df['yhat'][-int(periods):].sum()

        daily_data = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(int(periods)).copy()
        daily_data.rename(columns={'ds': 'date', 'yhat': 'prediction', 'yhat_lower': 'lower_bound', 'yhat_upper': 'upper_bound'}, inplace=True)
        daily_data['date'] = daily_data['date'].dt.strftime('%Y-%m-%d')

        response_data = {
            "daily_data": daily_data.to_dict('records'),
            "total_forecast": int(round(total_forecast))
        }
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": f"Failed to generate forecast: {e}"}), 500

# ADD THE FOLLOWING THREE FUNCTIONS TO app.py

@app.route('/api/recommend/accessories/instock', methods=['POST'])
def get_instock_accessory_recommendations():
    """Gets top accessory recommendations that are verified to be in stock."""
    if not recommender:
        return jsonify({"error": "Recommender system not loaded"}), 500
        
    data = request.get_json()
    phone_model_name = data.get('phone_model_name')
    store_id = data.get('store_id')

    if not phone_model_name or not store_id:
        return jsonify({"error": "phone_model_name and store_id are required"}), 400

    try:
        recommended_ids = recommender.recommend_accessories(phone_model_name=phone_model_name, top_n=5)
        if not recommended_ids:
            return jsonify([])
        
        # --- START OF DEFINITIVE FIX ---

        # Create named placeholders for the IN clause, e.g., ":p0, :p1, :p2"
        # This is the most reliable method for parameterizing IN clauses
        placeholders = ", ".join(f":p{i}" for i in range(len(recommended_ids)))
        
        # Dynamically build the final SQL query string with named placeholders
        instock_query_sql = f"SELECT product_id FROM accessory_inventory WHERE store_id = :sid AND quantity > 0 AND product_id IN ({placeholders})"
        
        # Build the corresponding dictionary of parameters
        params_dict = {"sid": int(store_id)}
        for i, pid in enumerate(recommended_ids):
            params_dict[f"p{i}"] = pid

        # Execute the query with the dictionary of parameters
        with db_engine.connect() as connection:
            instock_result = connection.execute(text(instock_query_sql), params_dict).fetchall()
        
        # --- END OF DEFINITIVE FIX ---

        instock_ids = {row[0] for row in instock_result}

        final_recommendations = [pid for pid in recommended_ids if pid in instock_ids][:3]
        if not final_recommendations:
            return jsonify([])

        recommended_products = products_df_global[products_df_global['product_id'].isin(final_recommendations)]
        # Preserve the recommendation order
        if not recommended_products.empty:
            recommended_products = recommended_products.set_index('product_id').loc[final_recommendations].reset_index()
        
        recommended_products.replace({np.nan: None}, inplace=True)
        
        return jsonify(recommended_products.to_dict('records'))
    except Exception as e:
        # Pass the original database error to the frontend for clearer debugging
        return jsonify({"error": f"Could not generate in-stock recommendations: {e}"}), 500

@app.route('/api/inventory/accessories/<int:store_id>', methods=['GET'])
def get_all_store_accessories(store_id):
    """Gets a list of all available (quantity > 0) accessories for a specific store."""
    try:
        query = text("""
            SELECT p.product_id, p.model_name, p.original_msrp
            FROM accessory_inventory ai
            JOIN products p ON ai.product_id = p.product_id
            WHERE ai.store_id = :sid AND ai.quantity > 0
            ORDER BY p.model_name;
        """)
        with db_engine.connect() as connection:
            results = connection.execute(query, {"sid": store_id}).fetchall()
        
        accessories = [{"product_id": row[0], "model_name": row[1], "price": row[2]} for row in results]
        return jsonify(accessories)
    except Exception as e:
        print(f"Error getting accessories for store {store_id}: {e}")
        return jsonify({"error": f"Could not retrieve accessories for store: {e}"}), 500

@app.route('/api/checkout', methods=['POST'])
@app.route('/api/checkout', methods=['POST'])
def process_checkout():
    data = request.get_json()
    cart_phone = data.get('phone')
    cart_accessories = data.get('accessories', [])
    store_id = data.get('storeId')

    if not cart_phone or not store_id:
        return jsonify({"error": "A phone and store ID are required for checkout."}), 400

    with db_engine.connect() as connection:
        transaction = connection.begin()
        try:
            # 1. Get new Transaction ID
            max_txn_id_result = connection.execute(text("SELECT MAX(transaction_id) FROM transactions")).scalar()
            new_txn_id = (max_txn_id_result or 0) + 1
            
            # --- FIX: Changed format to store only the date (YYYY-MM-DD) ---
            txn_date = datetime.now().strftime('%Y-%m-%d')

            # 2. Process Phone Sale
            phone_product_id = connection.execute(text("SELECT product_id FROM inventory_units WHERE unit_id = :uid"), {"uid": cart_phone['unit_id']}).scalar()
            
            # Insert phone transaction
            phone_txn_sql = text("INSERT INTO transactions (transaction_id, transaction_date, unit_id, product_id, store_id, final_sale_price) VALUES (:tid, :tdate, :uid, :pid, :sid, :price)")
            connection.execute(phone_txn_sql, {"tid": new_txn_id, "tdate": txn_date, "uid": cart_phone['unit_id'], "pid": phone_product_id, "sid": store_id, "price": cart_phone['price']})
            
            # Update phone inventory status
            update_phone_sql = text("UPDATE inventory_units SET status = 'Sold' WHERE unit_id = :uid")
            connection.execute(update_phone_sql, {"uid": cart_phone['unit_id']})
            
            # 3. Process Accessory Sales
            for acc in cart_accessories:
                # Insert accessory transaction (note: unit_id is NULL)
                acc_txn_sql = text("INSERT INTO transactions (transaction_id, transaction_date, product_id, store_id, final_sale_price) VALUES (:tid, :tdate, :pid, :sid, :price)")
                connection.execute(acc_txn_sql, {"tid": new_txn_id, "tdate": txn_date, "pid": acc['product_id'], "sid": store_id, "price": acc['price']})

                # Update accessory inventory quantity
                update_acc_sql = text("UPDATE accessory_inventory SET quantity = quantity - :qty WHERE product_id = :pid AND store_id = :sid")
                connection.execute(update_acc_sql, {"qty": acc['quantity'], "pid": acc['product_id'], "sid": store_id})

            # 4. Commit transaction
            transaction.commit()
            return jsonify({"success": True, "transaction_id": new_txn_id})

        except Exception as e:
            transaction.rollback()
            print(f"CHECKOUT FAILED: {e}")
            return jsonify({"error": f"An error occurred during checkout: {e}"}), 500

# 5. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    app.run(debug=True, port=5000)