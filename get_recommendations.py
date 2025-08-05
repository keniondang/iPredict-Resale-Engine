import pandas as pd
import os
import joblib

class Recommender:
    """
    Loads pre-built recommendation data and uses a hybrid scoring model
    to provide sophisticated accessory suggestions.
    """
    def __init__(self, data_path='models/recommendation_data_advanced.joblib', products_path='data/products.csv'):
        print("--- Initializing Advanced Recommendation Engine ---")
        try:
            recommendation_data = joblib.load(data_path)
            self.products_df = pd.read_csv(products_path)
        except FileNotFoundError as e:
            raise SystemExit(f"ERROR: {e}. Please run 'build_recommendations.py' first.")
            
        self.accessory_popularity = recommendation_data['popularity']
        self.compatibility_df = recommendation_data['compatibility']
        self.co_purchase_df = recommendation_data['co_purchase']
        
        # Prepare phone reference data
        self.phones_df = self.products_df[self.products_df['product_type'] == 'Used Phone'].copy()
        self.phones_df['base_model'] = self.phones_df['model_name'].str.extract(r'(iPhone \d{1,2}(?: Pro| Plus| Pro Max| Mini)?)')[0]
        
        print("Engine ready.")

    def recommend_accessories(self, phone_product_id: int, top_n: int = 3, popularity_weight: float = 0.4, copurchase_weight: float = 0.6) -> list:
        """
        Recommends accessories using a hybrid of overall popularity and co-purchase data.

        Args:
            phone_product_id (int): The product_id of the phone.
            top_n (int): The number of top accessories to recommend.
            popularity_weight (float): The weight for the overall popularity score.
            copurchase_weight (float): The weight for the co-purchase score.

        Returns:
            list: A list of the top_n recommended accessory product_ids.
        """
        # 1. Find the phone's base_model
        phone_info = self.phones_df[self.phones_df['product_id'] == phone_product_id]
        if phone_info.empty:
            print(f"Warning: Phone with product_id {phone_product_id} not found.")
            return []
        phone_base_model = phone_info['base_model'].iloc[0]
        
        # 2. Find all compatible accessories
        compatible = self.compatibility_df[self.compatibility_df['phone_product_id'] == phone_product_id]
        if compatible.empty:
            return []
        compatible_ids = compatible['accessory_product_id'].tolist()
        
        # 3. Calculate scores for each compatible accessory
        scores = {}
        for acc_id in compatible_ids:
            # Popularity Score (normalized)
            pop_score = self.accessory_popularity.get(acc_id, 0)
            
            # Co-purchase Score
            co_purchase_entry = self.co_purchase_df[
                (self.co_purchase_df['phone_base_model'] == phone_base_model) &
                (self.co_purchase_df['accessory_product_id'] == acc_id)
            ]
            co_purchase_score = co_purchase_entry['count'].iloc[0] if not co_purchase_entry.empty else 0
            
            scores[acc_id] = {'pop': pop_score, 'co': co_purchase_score}
            
        if not scores:
            return []

        scores_df = pd.DataFrame.from_dict(scores, orient='index')
        
        # Normalize scores from 0 to 1 to make them comparable
        if not scores_df['pop'].sum() == 0:
            scores_df['pop_norm'] = scores_df['pop'] / scores_df['pop'].max()
        else:
            scores_df['pop_norm'] = 0
            
        if not scores_df['co'].sum() == 0:
            scores_df['co_norm'] = scores_df['co'] / scores_df['co'].max()
        else:
            scores_df['co_norm'] = 0

        # 4. Calculate final hybrid score
        scores_df['final_score'] = (scores_df['pop_norm'] * popularity_weight) + (scores_df['co_norm'] * copurchase_weight)
        
        # 5. Sort by final score and get top N
        top_n_recommendations = scores_df.sort_values(by='final_score', ascending=False).head(top_n)
        
        return list(top_n_recommendations.index)

if __name__ == "__main__":
    recommender = Recommender()

    try:
        target_phone_name = 'iPhone 15 Pro 256GB'
        target_phone_id = recommender.phones_df[
            recommender.phones_df['model_name'] == target_phone_name
        ]['product_id'].iloc[0]
        
        print(f"\nGetting recommendations for: '{target_phone_name}' (ID: {target_phone_id})")

        recommended_ids = recommender.recommend_accessories(
            phone_product_id=target_phone_id,
            top_n=5
        )

        if recommended_ids:
            recommended_products = recommender.products_df[recommender.products_df['product_id'].isin(recommended_ids)]
            print("\nTop 3 Recommended Accessories (Hybrid Model):")
            for acc_id in recommended_ids: # Iterate in sorted order
                product_name = recommended_products[recommended_products['product_id'] == acc_id]['model_name'].iloc[0]
                print(f"  - {product_name} (Product ID: {acc_id})")
        else:
            print("\nCould not generate recommendations.")

    except IndexError:
        print(f"\nError: Could not find the phone '{target_phone_name}' in products.csv.")
