import pandas as pd
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import timedelta
from sqlalchemy import text

class DateFeatureCalculator(BaseEstimator, TransformerMixin):
    """
    Assumes the input DataFrame has already been merged with model spec data.
    Calculates age and seasonality features based on a reference date column
    that must be present in the DataFrame.
    """
    def __init__(self, reference_date_col):
        self.reference_date_col = reference_date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X_transformed = X.copy()
        X_transformed[self.reference_date_col] = pd.to_datetime(X_transformed[self.reference_date_col])
        X_transformed['release_date'] = pd.to_datetime(X_transformed['release_date'])
        X_transformed['successor_release_date'] = pd.to_datetime(X_transformed['successor_release_date'])

        X_transformed['days_since_model_release'] = (X_transformed[self.reference_date_col] - X_transformed['release_date']).dt.days
        X_transformed['days_since_successor_release'] = (X_transformed[self.reference_date_col] - X_transformed['successor_release_date']).dt.days
        X_transformed['has_successor'] = X_transformed['days_since_successor_release'].notna().astype(int)
        X_transformed['days_since_successor_release'] = X_transformed['days_since_successor_release'].fillna(0)

        X_transformed['month_of_year'] = X_transformed[self.reference_date_col].dt.month
        X_transformed['is_holiday_season'] = X_transformed['month_of_year'].isin([11, 12]).astype(int)

        return X_transformed

class MarketDemandCalculator(BaseEstimator, TransformerMixin):
    """
    Calculates market demand. This transformer must be fitted on the entire
    sales history to create an internal lookup mechanism.
    """
    def __init__(self, db_engine, lookback_days=7):
        self.db_engine = db_engine
        self.lookback_days = lookback_days

    def fit(self, X, y=None):
        # No pre-computation needed anymore, fit does nothing.
        return self

    def transform(self, X):

        X_transformed = X.copy()
        X_transformed['market_demand'] = X_transformed.apply(self._calculate_demand_for_row, axis=1)

        return X_transformed

    def _calculate_demand_for_row(self, row):
        reference_date = pd.to_datetime(row['transaction_date'])
        # Use product_id for specific demand signal
        product_id = row['product_id']
        lookback_start = reference_date - timedelta(days=self.lookback_days)

        # Query by product_id instead of base_model
        query = text("""
            SELECT COUNT(t.transaction_id)
            FROM transactions t
            WHERE t.product_id = :product_id
            AND t.transaction_date < :ref_date
            AND t.transaction_date >= :start_date
        """)

        with self.db_engine.connect() as connection:
            result = connection.execute(query, {
                'product_id': int(product_id),
                'ref_date': reference_date,
                'start_date': lookback_start
            }).scalar_one_or_none() or 0

        return result
    
    def __getstate__(self):
        # This method is called by pickle when saving the object.
        # We create a copy of the object's state and remove the unpicklable attribute.
        state = self.__dict__.copy()
        if 'db_engine' in state:
            del state['db_engine']
        return state

    def __setstate__(self, state):
        # This method is called by pickle when loading the object.
        # We restore the saved state and initialize the transient attribute to None.
        self.__dict__.update(state)
        self.db_engine = None

class GradeHistoryCalculator(BaseEstimator, TransformerMixin):
    """
    Calculates grade-specific historical features. Must be fitted on the
    full transaction history to pre-compute the features.
    """
    def fit(self, X, y=None):

        df = X.sort_values('transaction_date').copy()
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df_indexed = df.set_index('transaction_date')
        grouping_cols = ['product_id', 'grade']

        sales_counts = df_indexed.groupby(grouping_cols).rolling('90D')['unit_id'].count()
        avg_days = df_indexed.groupby(grouping_cols).rolling('90D')['days_in_stock'].mean()
        
        merged = pd.merge(
            sales_counts.reset_index(name='grade_specific_sales_last_90d'),
            avg_days.reset_index(name='grade_specific_avg_days_last_90d'),
            on=['transaction_date', 'product_id', 'grade'], how='outer'
        )
        
        merged['grade_specific_sales_last_90d'] = merged.groupby(grouping_cols)['grade_specific_sales_last_90d'].shift(1)
        merged['grade_specific_avg_days_last_90d'] = merged.groupby(grouping_cols)['grade_specific_avg_days_last_90d'].shift(1)
        
        self.latest_features_ = merged.sort_values('transaction_date').drop_duplicates(subset=grouping_cols, keep='last').set_index(grouping_cols)

        return self

    def transform(self, X):

        X_transformed = X.copy().set_index(['product_id', 'grade'])
        X_with_features = X_transformed.join(self.latest_features_[['grade_specific_sales_last_90d', 'grade_specific_avg_days_last_90d']])

        return X_with_features.reset_index().fillna(0)


class TargetDefiner(BaseEstimator, TransformerMixin):
    """
    Calculates 'days_in_stock' and defines the binary 'mover_category'
    target variable based on the median split learned during fit.
    """
    def fit(self, X, y=None):

        X_fit = X.copy()
        X_fit['transaction_date'] = pd.to_datetime(X_fit['transaction_date'])
        X_fit['acquisition_date'] = pd.to_datetime(X_fit['acquisition_date'])
        X_fit['days_in_stock'] = (X_fit['transaction_date'] - X_fit['acquisition_date']).dt.days
        X_fit = X_fit[X_fit['days_in_stock'] >= 0]
        
        self.median_days_in_stock_ = X_fit['days_in_stock'].median()
        print(f"TargetDefiner fitted. Median days in stock: {self.median_days_in_stock_:.0f}")

        return self

    def transform(self, X):

        X_transformed = X.copy()
        X_transformed['transaction_date'] = pd.to_datetime(X_transformed['transaction_date'])
        X_transformed['acquisition_date'] = pd.to_datetime(X_transformed['acquisition_date'])
        X_transformed['days_in_stock'] = (X_transformed['transaction_date'] - X_transformed['acquisition_date']).dt.days
        X_transformed['mover_category'] = np.where(X_transformed['days_in_stock'] <= self.median_days_in_stock_, 'Prime Inventory', 'Aging Inventory')

        return X_transformed