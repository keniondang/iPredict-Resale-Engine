import pandas as pd
import joblib
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for Matplotlib

# --- Import your custom classes and functions ---
from predict_price import PricePredictor
from predict_dynamic_price import DynamicPricePredictor
from predict_velocity import VelocityPredictor
from get_recommendations import Recommender
# Note: The logic from get_discontinuation_alerts and get_demand_forecast will be adapted directly into the API endpoints.

# -----------------------------------------------------------------------------
# FLASK APP INITIALIZATION AND CONFIGURATION
# -----------------------------------------------------------------------------

app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) to allow your React frontend
# (running on a different port) to communicate with this backend.
CORS(app)

# -----------------------------------------------------------------------------
# LOAD PREDICTOR CLASSES AND DATA AT STARTUP
# -----------------------------------------------------------------------------
# Instantiate the predictor classes. Each class handles loading its own models
# and necessary data files internally.

try:
    print("--- Initializing Predictor Classes and Data Engines ---")
    
    # Core prediction models
    price_predictor = PricePredictor(model_path='models/price_model.joblib')
    dynamic_price_predictor = DynamicPricePredictor(model_path='models/dynamic_selling_price_pipeline.joblib')
    velocity_predictor = VelocityPredictor(model_path='models/velocity_model.joblib')
    
    # Recommendation engine
    recommender = Recommender(data_path='models/recommendation_data_advanced.joblib')
    
    # Load data/models needed for other endpoints
    products_df = pd.read_csv('data/products.csv')
    discontinuation_list = joblib.load('models/discontinuation_list.joblib')

    print("--- All predictors and data loaded successfully. API is ready. ---")

except Exception as e:
    print(f"FATAL ERROR: Failed to initialize predictors or load data. Details: {e}")
    # In a real app, you might want to exit or have more robust error handling.
    price_predictor = None
    dynamic_price_predictor = None
    velocity_predictor = None
    recommender = None
    products_df = None
    discontinuation_list = None

# -----------------------------------------------------------------------------
# API ENDPOINTS
# -----------------------------------------------------------------------------

@app.route('/api/products', methods=['GET'])
def get_products():
    """Endpoint to get a list of all available phone models for frontend dropdowns."""
    if products_df is None:
        return jsonify({"error": "Products data not loaded"}), 500
    try:
        phone_models = products_df[products_df['product_type'] == 'Used Phone']['model_name'].unique().tolist()
        return jsonify(sorted(phone_models))
    except Exception as e:
        return jsonify({"error": f"Could not load products: {e}"}), 500


@app.route('/api/appraise/buy', methods=['POST'])
def appraise_buy():
    """
    Combined endpoint to predict both the buying price range and the sales velocity
    for a potential new acquisition.
    """
    if not price_predictor or not velocity_predictor:
        return jsonify({"error": "Predictor models not loaded"}), 500

    data = request.get_json()
    model_name = data.get('model_name')
    grade = data.get('grade')

    if not model_name or not grade:
        return jsonify({"error": "model_name and grade are required"}), 400

    appraisal_date_str = datetime.now().strftime('%Y-%m-%d')

    try:
        # Get predictions from both models
        price_result = price_predictor.predict(model_name, grade, appraisal_date_str)
        velocity_result = velocity_predictor.predict(model_name, grade, appraisal_date_str)

        if "error" in price_result:
            return jsonify(price_result), 404
        if "error" in velocity_result:
            return jsonify(velocity_result), 404

        # Combine results into a single, comprehensive response
        response = {
            "model_appraised": price_result.get("model_appraised"),
            "grade": price_result.get("grade"),
            "predicted_price_range": price_result.get("predicted_price_range"),
            "predicted_sales_velocity": velocity_result.get("predicted_sales_velocity")
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred during appraisal: {e}"}), 500


@app.route('/api/inventory/price', methods=['POST'])
def predict_dynamic_sell_price():
    """Endpoint to predict the dynamic selling price for an item already in inventory."""
    if not dynamic_price_predictor:
        return jsonify({"error": "Dynamic price predictor not loaded"}), 500

    data = request.get_json()
    unit_id = data.get('unit_id')
    if not unit_id:
        return jsonify({"error": "unit_id is required"}), 400

    prediction_date_str = datetime.now().strftime('%Y-%m-%d')

    try:
        result = dynamic_price_predictor.predict(int(unit_id), prediction_date_str)
        if "error" in result:
            return jsonify(result), 404
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


@app.route('/api/recommend/accessories', methods=['POST'])
def get_accessory_recommendations():
    """Endpoint to recommend accessories for a given phone."""
    if not recommender or products_df is None:
        return jsonify({"error": "Recommender system not loaded"}), 500
        
    data = request.get_json()
    phone_product_id = data.get('phone_product_id')
    if not phone_product_id:
        return jsonify({"error": "phone_product_id parameter is required"}), 400

    try:
        recommended_ids = recommender.recommend_accessories(phone_product_id=int(phone_product_id), top_n=3)
        
        if not recommended_ids:
            return jsonify([])
        
        # Get full product details for the recommended IDs
        recommended_products = products_df[products_df['product_id'].isin(recommended_ids)]
        
        # Ensure the order is preserved
        recommended_products = recommended_products.set_index('product_id').loc[recommended_ids].reset_index()
        
        return jsonify(recommended_products.to_dict('records'))
    except Exception as e:
        return jsonify({"error": f"Could not generate recommendations: {e}"}), 500


@app.route('/api/inventory/discontinuation_alerts', methods=['GET'])
def get_discontinuation_list():
    """Endpoint that returns the pre-built list of accessories to discontinue."""
    if discontinuation_list is None:
        return jsonify({"error": "Discontinuation list not loaded. Please run build_discontinuation_list.py."}), 500
    
    return jsonify(discontinuation_list)


@app.route('/api/demand_forecast', methods=['POST'])
def get_product_demand_forecast():
    """
    Loads a pre-trained Prophet model for a specific product and returns
    the forecast data as JSON, suitable for rendering a chart on the frontend.
    """
    data = request.get_json()
    category_value = data.get('product_name')
    periods = data.get('periods', 90)

    if not category_value:
        return jsonify({"error": "product_name parameter is required"}), 400

    # Sanitize the filename to match how it was saved
    safe_category_value = "".join(c for c in category_value if c.isalnum())
    model_path = os.path.join('models', f'demand_forecast_{safe_category_value}.joblib')

    try:
        model_package = joblib.load(model_path)
        model = model_package['model']
        release_date = model_package['release_date']
        successor_date = model_package['successor_date']
        base_cap = model_package['base_cap']
    except FileNotFoundError:
        return jsonify({"error": f"Forecast model for '{category_value}' not found. Please train it first."}), 404
    except Exception as e:
        return jsonify({"error": f"Could not load forecast model: {e}"}), 500

    try:
        # Create a dataframe for future dates and calculate the dynamic cap
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

        # Generate the forecast
        forecast_df = model.predict(future_df)
        
        # Prepare the data for JSON response, taking only the future forecast
        response_data = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(int(periods)).copy()
        response_data.rename(columns={
            'ds': 'date',
            'yhat': 'prediction',
            'yhat_lower': 'lower_bound',
            'yhat_upper': 'upper_bound'
        }, inplace=True)
        # Convert datetime to string for JSON compatibility
        response_data['date'] = response_data['date'].dt.strftime('%Y-%m-%d')
        
        return jsonify(response_data.to_dict('records'))
    except Exception as e:
        return jsonify({"error": f"Failed to generate forecast: {e}"}), 500

# -----------------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Runs the Flask app. `debug=True` allows for auto-reloading when you save changes.
    # In a production environment, you would use a proper WSGI server like Gunicorn.
    app.run(debug=True, port=5000)
