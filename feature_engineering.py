import pandas as pd
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import timedelta

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
        X_transformed['days_since_successor_release'] = X_transformed['days_since_successor_release'].fillna(-999)

        X_transformed['month_of_year'] = X_transformed[self.reference_date_col].dt.month
        X_transformed['is_holiday_season'] = X_transformed['month_of_year'].isin([11, 12]).astype(int)

        return X_transformed

class MarketDemandCalculator(BaseEstimator, TransformerMixin):
    """
    Calculates market demand. This transformer must be fitted on the entire
    sales history to create an internal lookup mechanism.
    """
    def __init__(self, lookback_days=7):
        self.lookback_days = lookback_days

    def fit(self, X, y=None):

        self.sales_history_ = X.copy()
        self.sales_history_['transaction_date'] = pd.to_datetime(self.sales_history_['transaction_date'])
        self.unique_base_models_ = set(self.sales_history_['base_model'].unique())

        return self

    def transform(self, X):

        X_transformed = X.copy()
        X_transformed['market_demand'] = X_transformed.apply(self._calculate_demand_for_row, axis=1)

        return X_transformed

    def _get_predecessor(self, base_model):

        if not isinstance(base_model, str): return None
        match = re.search(r'(\d+)', base_model)

        if not match: return None
        pred_version = int(match.group(1)) - 1

        return base_model.replace(str(match.group(1)), str(pred_version))

    def _calculate_demand_for_row(self, row):

        reference_date = pd.to_datetime(row['transaction_date'])
        base_model = row['base_model']
        
        history = self.sales_history_
        lookback_start = reference_date - timedelta(days=self.lookback_days)

        demand = history[
            (history['base_model'] == base_model) &
            (history['transaction_date'] < reference_date) &
            (history['transaction_date'] >= lookback_start)
        ].shape[0]

        if demand == 0:
            predecessor = self._get_predecessor(base_model)
            if predecessor and predecessor in self.unique_base_models_:
                demand = history[
                    (history['base_model'] == predecessor) &
                    (history['transaction_date'] < reference_date) &
                    (history['transaction_date'] >= lookback_start)
                ].shape[0]

        return demand

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

        sales_counts = df_indexed.groupby(grouping_cols).rolling('120D')['unit_id'].count()
        avg_days = df_indexed.groupby(grouping_cols).rolling('120D')['days_in_stock'].mean()
        
        merged = pd.merge(
            sales_counts.reset_index(name='grade_specific_sales_last_120d'),
            avg_days.reset_index(name='grade_specific_avg_days_last_120d'),
            on=['transaction_date', 'product_id', 'grade'], how='outer'
        )
        
        merged['grade_specific_sales_last_120d'] = merged.groupby(grouping_cols)['grade_specific_sales_last_120d'].shift(1)
        merged['grade_specific_avg_days_last_120d'] = merged.groupby(grouping_cols)['grade_specific_avg_days_last_120d'].shift(1)
        
        self.latest_features_ = merged.sort_values('transaction_date').drop_duplicates(subset=grouping_cols, keep='last').set_index(grouping_cols)

        return self

    def transform(self, X):

        X_transformed = X.copy().set_index(['product_id', 'grade'])
        X_with_features = X_transformed.join(self.latest_features_[['grade_specific_sales_last_120d', 'grade_specific_avg_days_last_120d']])

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