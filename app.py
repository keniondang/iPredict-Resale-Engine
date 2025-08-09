import pandas as pd
import numpy as np
import joblib
import traceback
from sqlalchemy import text
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from sqlalchemy import create_engine
import urllib

SERVER_NAME = "localhost\\SQLEXPRESS"
DATABASE_NAME = "UsedPhoneResale"

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

    def recommend_accessories(self, phone_model_name: str, top_n: int = 3, pop_weight: float = 0.2, copurchase_weight: float = 0.8):
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

app = Flask(__name__)
CORS(app)
db_engine = get_db_engine()

try:
    print("--- Initializing: Loading Data and Pre-trained Pipelines ---")
    products_df_global = pd.read_sql("SELECT * FROM products", db_engine)
    products_df_global['base_model'] = products_df_global['model_name'].str.extract(r'(iPhone \d{1,2}(?: Pro| Plus| Pro Max| Mini)?)')[0]
    
    stores_df_global = pd.read_sql("SELECT * FROM stores", db_engine)
    
    price_model_artifact = joblib.load('models/price_model_pipeline.joblib')
    velocity_model_artifact = joblib.load('models/velocity_model_pipeline.joblib')
    dynamic_price_artifact = joblib.load('models/dynamic_selling_price_pipeline.joblib')

    recommender = Recommender('models/recommendation_data.joblib', products_df_global)
    discontinuation_list_global = joblib.load('models/discontinuation_list.joblib')
    discontinuation_id_set = {item['product_id'] for item in discontinuation_list_global}
    
    print("--- All pipelines and data loaded successfully. API is ready. ---")

except Exception as e:
    print(f"FATAL ERROR: Failed to initialize. Details: {e}")
    price_model_artifact = velocity_model_artifact = dynamic_price_artifact = None
    recommender = discontinuation_list_global = discontinuation_id_set = None


@app.route('/api/products', methods=['GET'])
def get_products():

    phone_models = products_df_global[products_df_global['product_type'] == 'Used Phone']['model_name'].unique().tolist()

    return jsonify(sorted(phone_models))

@app.route('/api/stores', methods=['GET'])
def get_stores():

    physical_stores = stores_df_global[stores_df_global['store_name'] != 'Online Store']
    stores_list = physical_stores[['store_id', 'store_name']].to_dict('records')

    return jsonify(sorted(stores_list, key=lambda x: x['store_name']))

@app.route('/api/appraise/buy', methods=['POST'])
def appraise_buy():

    if not all([price_model_artifact, velocity_model_artifact, dynamic_price_artifact]):

        return jsonify({"error": "Predictor models not loaded"}), 500

    data = request.get_json()
    model_name = data.get('model_name')
    grade = data.get('grade')
    desired_profit = float(data.get('desired_profit', 0))

    phone_info = products_df_global[products_df_global['model_name'] == model_name]
    
    if phone_info.empty:
        return jsonify({"error": f"Model '{model_name}' not found."}), 404
        
    appraisal_df = pd.DataFrame([{
        'product_id': phone_info.iloc[0]['product_id'],
        'model_name': model_name,
        'base_model': phone_info.iloc[0]['base_model'],
        'model_tier': phone_info.iloc[0]['model_tier'],
        'storage_gb': phone_info.iloc[0]['storage_gb'],
        'original_msrp': phone_info.iloc[0]['original_msrp'],
        'grade': grade,
        'release_date': phone_info.iloc[0]['release_date'],
        'successor_release_date': phone_info.iloc[0]['successor_release_date'],
        'acquisition_date': datetime.now(),
        'transaction_date': datetime.now()
    }])

    price_predictions = {
        name: round(pipe.predict(appraisal_df)[0], 2)
        for name, pipe in price_model_artifact['models'].items()
    }
    
    velocity_prediction = velocity_model_artifact['pipeline'].predict(appraisal_df)[0]
    
    selling_price_predictions = {
        name: round(pipe.predict(appraisal_df)[0], 2)
        for name, pipe in dynamic_price_artifact['models'].items()
    }

    profit_target_range = {
        "low": selling_price_predictions.get('low_liquidate_price', 0) - desired_profit,
        "median": selling_price_predictions.get('median_fair_market_price', 0) - desired_profit,
        "high": selling_price_predictions.get('high_start_price', 0) - desired_profit,
    }
    
    market_alert = bool(price_predictions.get("median", 0) > profit_target_range["high"])

    return jsonify({
        "model_appraised": model_name,
        "grade": grade,
        "historically_based_buy_price": price_predictions,
        "predicted_sales_velocity": str(velocity_prediction),
        "predicted_selling_price_range": selling_price_predictions,
        "profit_targeted_buy_price_range": {k: round(v, 2) for k, v in profit_target_range.items()},
        "market_alert": market_alert
    })

@app.route('/api/inventory/buy', methods=['POST'])
def buy_inventory():

    data = request.get_json()
    model_name, grade, store_id, acquisition_price = data.get('model_name'), data.get('grade'), data.get('store_id'), data.get('acquisition_price')

    if not all([model_name, grade, store_id, acquisition_price]):
        return jsonify({"error": "All fields are required"}), 400

    try:
        product_details = products_df_global[products_df_global['model_name'] == model_name].iloc[0]
        product_id = product_details['product_id']
        release_date = pd.to_datetime(product_details['release_date'])
        acquisition_date = datetime.now()
        age_at_acquisition = (acquisition_date - release_date).days
        
        with db_engine.connect() as connection:
            transaction = connection.begin()
            try:
                max_unit_id_result = connection.execute(text("SELECT MAX(unit_id) FROM inventory_units")).scalar()
                new_unit_id = (max_unit_id_result or 0) + 1
                insert_query = text("""
                    INSERT INTO inventory_units (unit_id, product_id, store_id, acquisition_date, grade, age_at_acquisition_days, acquisition_price, status)
                    VALUES (:unit_id, :product_id, :store_id, :acquisition_date, :grade, :age_at_acquisition_days, :acquisition_price, 'In Stock')
                """)
                connection.execute(insert_query, {
                    "unit_id": new_unit_id, "product_id": int(product_id), "store_id": int(store_id),
                    "acquisition_date": acquisition_date.strftime('%Y-%m-%d'), "grade": grade,
                    "age_at_acquisition_days": age_at_acquisition, "acquisition_price": float(acquisition_price)
                })
                transaction.commit()

                return jsonify({"success": True, "unit_id": new_unit_id})
            
            except Exception as e:
                transaction.rollback()
                return jsonify({"error": f"Database error: {e}"}), 500
            
    except IndexError:
        return jsonify({"error": f"Model '{model_name}' not found."}), 404
    
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


@app.route('/api/inventory/price', methods=['POST'])
def predict_dynamic_sell_price():

    if not dynamic_price_artifact:
        return jsonify({"error": "Dynamic price predictor not loaded"}), 500
        
    data = request.get_json()
    unit_id = int(data.get('unit_id'))

    with db_engine.connect() as conn:
        unit_info = pd.read_sql(text("SELECT * FROM inventory_units WHERE unit_id = :uid"), conn, params={"uid": unit_id})
    
    if unit_info.empty:
        return jsonify({"error": f"Unit ID {unit_id} not found."}), 404
        
    full_details = pd.merge(unit_info, products_df_global, on='product_id')
    full_details['transaction_date'] = datetime.now()

    predictions = {
        name: round(pipe.predict(full_details)[0], 2)
        for name, pipe in dynamic_price_artifact['models'].items()
    }
    
    unit_data = full_details.iloc[0].to_dict()
    
    response_data = {
        "unit_id": int(unit_data['unit_id']),
        "model_name": unit_data['model_name'],
        "status": unit_data['status'],
        "days_in_inventory": int((datetime.now() - pd.to_datetime(unit_data['acquisition_date'])).days) if pd.notna(unit_data['acquisition_date']) else None,
        "acquisition_date": pd.to_datetime(unit_data['acquisition_date']).strftime('%Y-%m-%d') if pd.notna(unit_data['acquisition_date']) else None,
        "acquisition_price": float(unit_data['acquisition_price']) if pd.notna(unit_data['acquisition_price']) else None,
        "predicted_selling_price_range": predictions
    }

    return jsonify(response_data)

@app.route('/api/inventory/combined', methods=['GET'])
def get_combined_inventory():

    try:
        with db_engine.connect() as connection:
            phones_df = pd.read_sql(text("""
                SELECT i.unit_id, i.grade, i.acquisition_date, i.store_id, p.* FROM inventory_units i
                JOIN products p ON i.product_id = p.product_id WHERE i.status = 'In Stock'
            """), connection)
            accessories_df = pd.read_sql(text("""
                SELECT ai.product_id, ai.quantity, p.model_name, s.store_name, ai.store_id FROM accessory_inventory ai
                JOIN products p ON ai.product_id = p.product_id JOIN stores s ON ai.store_id = s.store_id
                WHERE ai.quantity > 0
            """), connection)

        if not phones_df.empty:
            phones_df['base_model'] = phones_df['model_name'].str.extract(r'(iPhone \d{1,2}(?: Pro| Plus| Pro Max| Mini)?)')[0]
            phones_df['velocity_label'] = velocity_model_artifact['pipeline'].predict(phones_df)
        
        if not accessories_df.empty:
            accessories_df['discontinuation_label'] = accessories_df['product_id'].apply(
                lambda x: 'Discontinuation Alert' if x in discontinuation_id_set else None)

        response_data = {s['store_name']: {"summary": {}, "phones": [], "accessories": []} for _, s in stores_df_global[stores_df_global['location_type'] == 'Physical Store'].iterrows()}

        for store_name, group in accessories_df.groupby('store_name'):
            if store_name in response_data:
                response_data[store_name]['accessories'] = group.to_dict('records')
        
        phones_with_stores = pd.merge(phones_df, stores_df_global[['store_id', 'store_name']], on='store_id')

        for store_name, group in phones_with_stores.groupby('store_name'):
            if store_name in response_data:
                clean_group = group.replace({np.nan: None, pd.NaT: None})
                response_data[store_name]['phones'] = clean_group.to_dict('records')
        
        for store in response_data:
            response_data[store]['summary']['phone_units'] = len(response_data[store]['phones'])
            response_data[store]['summary']['accessory_types'] = len(response_data[store]['accessories'])

        return jsonify(response_data)
    
    except Exception as e:
        traceback.print_exc()

        return jsonify({"error": f"Could not load combined inventory: {e}"}), 500

@app.route('/api/recommend/accessories', methods=['POST'])
def get_accessory_recommendations():

    data = request.get_json()
    phone_model_name = data.get('phone_model_name')
    recommended_ids = recommender.recommend_accessories(phone_model_name=phone_model_name, top_n=3)
    if not recommended_ids: return jsonify([])
    
    recommended_products = products_df_global[products_df_global['product_id'].isin(recommended_ids)]
    ordered_products = recommended_products.set_index('product_id').loc[recommended_ids].reset_index()
    

    clean_products = ordered_products.replace({np.nan: None, pd.NaT: None})
    
    return jsonify(clean_products.to_dict('records'))


@app.route('/api/inventory/discontinuation_alerts', methods=['GET'])
def get_discontinuation_list():
    return jsonify(discontinuation_list_global)


@app.route('/api/demand_forecast', methods=['POST'])
def get_product_demand_forecast():

    data = request.get_json()
    product_name, periods = data.get('product_name'), data.get('periods', 90)
    if not product_name: return jsonify({"error": "product_name is required"}), 400
    
    safe_name = "".join(c for c in product_name if c.isalnum())
    model_path = os.path.join('models', f'demand_forecast_{safe_name}.joblib')
    
    try:
        model_package = joblib.load(model_path)
    except FileNotFoundError:
        return jsonify({"error": f"Forecast model for '{product_name}' not found"}), 404
    
    model, release_date, successor_date, base_cap = model_package.values()
    future_df = model.make_future_dataframe(periods=int(periods))
    
    def calculate_cap(row_date):
        decay = 1 - ((row_date - release_date).days / 730)
        if pd.notna(successor_date) and row_date > successor_date:
            decay = min(decay, 0.5 - ((row_date - successor_date).days / 365))
        return max(0.01, base_cap * decay)
        
    future_df['cap'], future_df['floor'] = future_df['ds'].apply(calculate_cap), 0
    forecast = model.predict(future_df).tail(int(periods))
    total_forecast = forecast['yhat'].sum()

    return jsonify({
        "daily_data": forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={'ds': 'date', 'yhat': 'prediction', 'yhat_lower': 'lower_bound', 'yhat_upper': 'upper_bound'}).to_dict('records'),
        "total_forecast": int(round(max(0, total_forecast)))
    })

@app.route('/api/checkout', methods=['POST'])
def process_checkout():
    data = request.get_json()
    cart_phone, cart_accessories, store_id = data.get('phone'), data.get('accessories', []), data.get('storeId')

    if not cart_phone or not store_id:
        return jsonify({"error": "A phone and store ID are required for checkout."}), 400

    with db_engine.connect() as connection:
        transaction = connection.begin()
        try:
            max_txn_id = (connection.execute(text("SELECT MAX(transaction_id) FROM transactions")).scalar() or 0) + 1
            txn_date = datetime.now().strftime('%Y-%m-%d')
            phone_pid = connection.execute(text("SELECT product_id FROM inventory_units WHERE unit_id = :uid"), {"uid": cart_phone['unit_id']}).scalar()
            
            connection.execute(text("INSERT INTO transactions (transaction_id, transaction_date, unit_id, product_id, store_id, final_sale_price) VALUES (:tid, :tdate, :uid, :pid, :sid, :price)"), 
                               {"tid": max_txn_id, "tdate": txn_date, "uid": cart_phone['unit_id'], "pid": int(phone_pid), "sid": store_id, "price": cart_phone['price']})
            connection.execute(text("UPDATE inventory_units SET status = 'Sold' WHERE unit_id = :uid"), {"uid": cart_phone['unit_id']})
            
            for acc in cart_accessories:
                connection.execute(text("INSERT INTO transactions (transaction_id, transaction_date, product_id, store_id, final_sale_price) VALUES (:tid, :tdate, :pid, :sid, :price)"),
                                   {"tid": max_txn_id, "tdate": txn_date, "pid": acc['product_id'], "sid": store_id, "price": acc['price']})
                connection.execute(text("UPDATE accessory_inventory SET quantity = quantity - :qty WHERE product_id = :pid AND store_id = :sid"),
                                   {"qty": acc['quantity'], "pid": acc['product_id'], "sid": store_id})
            transaction.commit()

            return jsonify({"success": True, "transaction_id": max_txn_id})
        
        except Exception as e:
            transaction.rollback()

            return jsonify({"error": f"An error occurred during checkout: {e}"}), 500

@app.route('/api/accessories/details/<int:product_id>', methods=['GET'])
def get_accessory_details(product_id):

    query = text("SELECT model_name, original_msrp FROM products WHERE product_id = :pid")
    compat_query = text("""SELECT p.model_name FROM accessory_compatibility ac JOIN products p ON ac.phone_product_id = p.product_id
                           WHERE ac.accessory_product_id = :pid ORDER BY p.release_date DESC;""")
    
    with db_engine.connect() as connection:
        info = connection.execute(query, {"pid": product_id}).fetchone()
        if not info: return jsonify({"error": "Accessory not found"}), 404
        phones = [row[0] for row in connection.execute(compat_query, {"pid": product_id}).fetchall()]

    return jsonify({"product_id": product_id, "model_name": info[0], "original_msrp": info[1], "compatible_phones": phones})

@app.route('/api/transactions/history', methods=['GET'])
def get_transaction_history():

    try:
        query = text("""
            SELECT t.transaction_id, t.transaction_date, p.model_name, p.product_type, i.grade, t.final_sale_price,
                   i.acquisition_price AS cost, s.store_name, s.store_id, i.acquisition_date
            FROM transactions t JOIN inventory_units i ON t.unit_id = i.unit_id JOIN products p ON i.product_id = p.product_id
            JOIN stores s ON i.store_id = s.store_id WHERE p.product_type = 'Used Phone'
            UNION ALL
            SELECT t.transaction_id, t.transaction_date, p.model_name, p.product_type, NULL AS grade, t.final_sale_price,
                   p.original_msrp AS cost, s.store_name, s.store_id, NULL as acquisition_date
            FROM transactions t JOIN products p ON t.product_id = p.product_id JOIN stores s ON t.store_id = s.store_id
            WHERE p.product_type = 'Accessory'
        """)
        with db_engine.connect() as connection: df = pd.read_sql_query(query, connection)

        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        if request.args.get('start_date'): df = df[df['transaction_date'] >= pd.to_datetime(request.args.get('start_date'))]
        if request.args.get('end_date'): df = df[df['transaction_date'] <= pd.to_datetime(request.args.get('end_date'))]
        if request.args.get('model_name'): df = df[df['model_name'] == request.args.get('model_name')]
        if request.args.get('grade'): df = df[df['grade'] == request.args.get('grade')]
        if request.args.get('store_id'): df = df[df['store_id'] == int(request.args.get('store_id'))]

        if df.empty: return jsonify({"summary": {}, "transactions": []})
        
        df['acquisition_date'] = pd.to_datetime(df['acquisition_date'], errors='coerce')
        df['profit_loss'] = df['final_sale_price'] - df['cost']
        df['days_in_stock'] = (df['transaction_date'] - df['acquisition_date']).dt.days
        phones_df = df[df['product_type'] == 'Used Phone'].copy()
        
        summary = {
            "total_revenue": float(df['final_sale_price'].sum()), "total_cost": float(df['cost'].sum()),
            "total_profit": float(df['profit_loss'].sum()), "total_items_sold": int(len(df)),
            "average_profit_per_item": float(df['profit_loss'].mean()), "average_days_in_stock": float(phones_df['days_in_stock'].mean())
        }
        
        df['transaction_date'] = df['transaction_date'].dt.strftime('%Y-%m-%d')
        df.replace({np.nan: None, pd.NaT: None, np.inf: None, -np.inf: None}, inplace=True)

        return jsonify({"summary": summary, "transactions": df.to_dict('records')})
    
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

@app.route('/api/recommend/accessories/instock', methods=['POST'])
def get_instock_accessory_recommendations():

    data = request.get_json()
    phone_model_name, store_id = data.get('phone_model_name'), data.get('store_id')
    recommended_ids = recommender.recommend_accessories(phone_model_name=phone_model_name, top_n=5)
    if not recommended_ids: return jsonify([])
    
    placeholders = ", ".join([f":p{i}" for i in range(len(recommended_ids))])
    params = {"sid": int(store_id), **{f"p{i}": pid for i, pid in enumerate(recommended_ids)}}
    
    with db_engine.connect() as connection:
        instock_ids = {row[0] for row in connection.execute(text(f"SELECT product_id FROM accessory_inventory WHERE store_id = :sid AND quantity > 0 AND product_id IN ({placeholders})"), params).fetchall()}

    final_recommendations = [pid for pid in recommended_ids if pid in instock_ids][:3]
    if not final_recommendations: return jsonify([])

    recommended_products = products_df_global[products_df_global['product_id'].isin(final_recommendations)]
    ordered_products = recommended_products.set_index('product_id').loc[final_recommendations].reset_index()
    clean_products = ordered_products.replace({np.nan: None, pd.NaT: None})

    return jsonify(clean_products.to_dict('records'))

@app.route('/api/inventory/accessories/<int:store_id>', methods=['GET'])
def get_all_store_accessories(store_id):

    query = text("SELECT p.product_id, p.model_name, p.original_msrp as price FROM accessory_inventory ai JOIN products p ON ai.product_id = p.product_id WHERE ai.store_id = :sid AND ai.quantity > 0 ORDER BY p.model_name;")
    with db_engine.connect() as connection:
        results = connection.execute(query, {"sid": store_id}).fetchall()
        
    return jsonify([dict(row._mapping) for row in results])

if __name__ == '__main__':
    app.run(debug=True, port=5000)