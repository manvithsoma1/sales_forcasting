"""
Feature Engineering Module — v2.0
Creates 40+ features: time, holiday, oil, transaction, lag, and rolling statistics.
"""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """Builds rich feature sets for sales forecasting models."""

    def __init__(self):
        self.lag_days = [1, 7, 14, 21, 28]
        self.rolling_windows = [7, 14, 30]

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def create_features(self, df, holidays_df=None, include_lags=True):
        """
        Master feature builder.

        Parameters
        ----------
        df : DataFrame with at least 'date' column. 
             Should already have: sales, onpromotion, dcoilwtico, 
             transactions, store_nbr, family, city, cluster, state, type.
        holidays_df : raw holidays_events DataFrame (optional, for detailed holiday features).
        include_lags : bool, whether to add lag/rolling features (requires store_nbr & family columns).

        Returns
        -------
        DataFrame with all engineered features.
        """
        print("Engineering features (v2.0)...")
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Ensure required base columns exist
        self._ensure_columns(df)

        # Feature groups
        df = self._time_features(df)
        df = self._oil_features(df)
        df = self._transaction_features(df)

        if holidays_df is not None:
            df = self._holiday_features(df, holidays_df)
        else:
            df = self._simple_holiday_features(df)

        if include_lags and 'store_nbr' in df.columns and 'family' in df.columns:
            df = self._lag_features(df)
            df = self._rolling_features(df)

        print(f"Features created: {len(df.columns)} columns, {len(df)} rows")
        return df

    def get_feature_columns(self, mode='lgbm'):
        """
        Returns the list of feature column names (excludes target and identifiers).
        
        Parameters
        ----------
        mode : 'lgbm' | 'lstm' | 'all'
        """
        base = [
            'onpromotion', 'dcoilwtico', 'is_holiday',
            'day_of_week', 'day_of_month', 'week_of_year', 'month', 'quarter',
            'is_month_start', 'is_month_end', 'is_quarter_end', 'is_payday',
            'is_national_holiday', 'is_regional_holiday',
            'days_to_next_holiday', 'days_since_last_holiday',
            'oil_price_7d_ma', 'oil_price_30d_ma', 'oil_price_volatility', 'oil_price_trend',
            'txn_7d_ma', 'txn_deviation', 'transactions',
        ]

        lag_cols = [f'sales_lag_{d}' for d in self.lag_days]
        roll_cols = []
        for w in self.rolling_windows:
            roll_cols += [f'sales_roll_mean_{w}', f'sales_roll_std_{w}']

        if mode == 'lstm':
            # LSTM uses a subset (no categoricals, no lags — handled via sequences)
            return base
        elif mode == 'lgbm':
            return base + lag_cols + roll_cols
        else:
            return base + lag_cols + roll_cols

    # ------------------------------------------------------------------
    # PRIVATE — Time Features
    # ------------------------------------------------------------------

    def _time_features(self, df):
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)

        # Payday — Ecuador pays on 15th and last day of month
        df['is_payday'] = (
            (df['day_of_month'] == 15) | (df['is_month_end'] == 1)
        ).astype(int)

        return df

    # ------------------------------------------------------------------
    # PRIVATE — Holiday Features
    # ------------------------------------------------------------------

    def _holiday_features(self, df, holidays_df):
        """Detailed holiday features from raw holidays_events.csv."""
        hol = holidays_df.copy()
        hol['date'] = pd.to_datetime(hol['date'])
        hol = hol[hol['transferred'] == False].copy()

        # National holidays
        national = hol[hol['locale'] == 'National']['date'].unique()
        df['is_national_holiday'] = df['date'].isin(national).astype(int)

        # Regional holidays
        regional = hol[hol['locale'] == 'Regional']['date'].unique()
        df['is_regional_holiday'] = df['date'].isin(regional).astype(int)

        # Days to next / since last holiday
        all_holiday_dates = sorted(pd.to_datetime(hol['date'].unique()))
        df['days_to_next_holiday'] = df['date'].apply(
            lambda d: self._days_to_nearest(d, all_holiday_dates, direction='next')
        )
        df['days_since_last_holiday'] = df['date'].apply(
            lambda d: self._days_to_nearest(d, all_holiday_dates, direction='prev')
        )

        # Ensure is_holiday exists
        if 'is_holiday' not in df.columns:
            df['is_holiday'] = (df['is_national_holiday'] | df['is_regional_holiday']).astype(int)

        return df

    def _simple_holiday_features(self, df):
        """Fallback when no holidays_df provided — fills with defaults."""
        for col, default in [
            ('is_national_holiday', 0), ('is_regional_holiday', 0),
            ('days_to_next_holiday', 30), ('days_since_last_holiday', 30)
        ]:
            if col not in df.columns:
                df[col] = default
        if 'is_holiday' not in df.columns:
            df['is_holiday'] = 0
        return df

    @staticmethod
    def _days_to_nearest(date, holiday_dates, direction='next'):
        """Calculate days to next or since last holiday."""
        if direction == 'next':
            future = [h for h in holiday_dates if h >= date]
            return (future[0] - date).days if future else 30
        else:
            past = [h for h in holiday_dates if h <= date]
            return (date - past[-1]).days if past else 30

    # ------------------------------------------------------------------
    # PRIVATE — Oil Price Features
    # ------------------------------------------------------------------

    def _oil_features(self, df):
        if 'dcoilwtico' not in df.columns:
            df['dcoilwtico'] = 0.0

        # Forward-fill missing oil prices
        df['dcoilwtico'] = df['dcoilwtico'].ffill().bfill().fillna(0)

        df['oil_price_7d_ma'] = df['dcoilwtico'].rolling(7, min_periods=1).mean()
        df['oil_price_30d_ma'] = df['dcoilwtico'].rolling(30, min_periods=1).mean()
        df['oil_price_volatility'] = df['dcoilwtico'].rolling(7, min_periods=1).std().fillna(0)
        df['oil_price_trend'] = df['dcoilwtico'].diff(7).fillna(0)

        return df

    # ------------------------------------------------------------------
    # PRIVATE — Transaction Features
    # ------------------------------------------------------------------

    def _transaction_features(self, df):
        if 'transactions' not in df.columns:
            df['transactions'] = 0

        df['transactions'] = df['transactions'].fillna(0)
        df['txn_7d_ma'] = df['transactions'].rolling(7, min_periods=1).mean()
        df['txn_deviation'] = np.where(
            df['txn_7d_ma'] > 0,
            df['transactions'] / df['txn_7d_ma'],
            1.0
        )
        return df

    # ------------------------------------------------------------------
    # PRIVATE — Lag & Rolling Features
    # ------------------------------------------------------------------

    def _lag_features(self, df):
        """Per store×family lag features to avoid data leakage."""
        if 'sales' not in df.columns:
            return df

        group_cols = ['store_nbr', 'family']
        # Verify group columns exist
        if not all(c in df.columns for c in group_cols):
            # Fall back to global lags
            for lag in self.lag_days:
                df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
            return df

        for lag in self.lag_days:
            df[f'sales_lag_{lag}'] = df.groupby(group_cols)['sales'].shift(lag)

        return df

    def _rolling_features(self, df):
        """Per store×family rolling stats (shifted by 1 to prevent leakage)."""
        if 'sales' not in df.columns:
            return df

        group_cols = ['store_nbr', 'family']
        if not all(c in df.columns for c in group_cols):
            for window in self.rolling_windows:
                shifted = df['sales'].shift(1)
                df[f'sales_roll_mean_{window}'] = shifted.rolling(window, min_periods=1).mean()
                df[f'sales_roll_std_{window}'] = shifted.rolling(window, min_periods=1).std().fillna(0)
            return df

        for window in self.rolling_windows:
            df[f'sales_roll_mean_{window}'] = df.groupby(group_cols)['sales'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            df[f'sales_roll_std_{window}'] = df.groupby(group_cols)['sales'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0)
            )

        return df

    # ------------------------------------------------------------------
    # PRIVATE — Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_columns(df):
        """Ensure required columns exist with sensible defaults."""
        defaults = {
            'sales': 0,
            'onpromotion': 0,
            'dcoilwtico': 0.0,
            'is_holiday': 0,
            'transactions': 0
        }
        for col, default in defaults.items():
            if col not in df.columns:
                print(f"⚠️ '{col}' missing — filling with {default}")
                df[col] = default
