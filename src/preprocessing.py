"""
Preprocessing Module — v2.0
Supports LSTM sequences, LightGBM tabular, and TFT dataset preparation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yaml


class Preprocessor:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.look_back = self.config['model']['look_back_days']

    # ------------------------------------------------------------------
    # LSTM path — scale + sequence
    # ------------------------------------------------------------------

    def scale_data(self, df, fit=True):
        """
        Scales numeric features using MinMaxScaler.
        Ensures 'sales' is the first column.

        Parameters
        ----------
        df : DataFrame with numeric columns (target = 'sales')
        fit : bool, True to fit_transform, False to transform only

        Returns
        -------
        (df_scaled, scaler)
        """
        print("Scaling data...")
        df_numeric = df.select_dtypes(include=[np.number]).copy()

        if df_numeric.isna().any().any():
            print("[WARNING] Found NaNs — filling with 0.")
            df_numeric = df_numeric.fillna(0)

        if 'sales' not in df_numeric.columns:
            raise ValueError("Expected 'sales' column not found.")

        # Ensure sales is first column
        cols = ['sales'] + [c for c in df_numeric.columns if c != 'sales']
        df_numeric = df_numeric[cols]

        if fit:
            scaled = self.scaler.fit_transform(df_numeric)
        else:
            scaled = self.scaler.transform(df_numeric)

        df_scaled = pd.DataFrame(scaled, columns=df_numeric.columns)
        return df_scaled, self.scaler

    def create_sequences(self, data):
        """
        Converts data into 3D sequences for LSTM: (Samples, TimeSteps, Features).
        Input: past 'look_back' days → Output: sales on next day.
        """
        print("Creating LSTM sequences...")
        X, y = [], []
        for i in range(len(data) - self.look_back):
            X.append(data[i: i + self.look_back])
            y.append(data[i + self.look_back, 0])  # 0 = sales column
        return np.array(X), np.array(y)

    # ------------------------------------------------------------------
    # LightGBM path — flat tabular data
    # ------------------------------------------------------------------

    def prepare_lgbm_data(self, df, feature_cols, target_col='sales'):
        """
        Prepare flat tabular data for LightGBM.

        Parameters
        ----------
        df : DataFrame with all features
        feature_cols : list of feature column names
        target_col : target column name

        Returns
        -------
        (X, y) — feature matrix and target array
        """
        print("Preparing LightGBM data...")

        # Filter to available columns
        available = [c for c in feature_cols if c in df.columns]
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            print(f"[WARNING] Missing columns skipped: {missing}")

        X = df[available].copy()
        y = df[target_col].copy() if target_col in df.columns else None

        # Convert categorical columns
        cat_cols = self.config.get('model', {}).get('lightgbm', {}).get('categorical_features', [])
        for col in cat_cols:
            if col in X.columns:
                X[col] = X[col].astype('category')

        # Fill NaNs
        X = X.fillna(0)
        if y is not None:
            y = y.fillna(0)

        return X, y

    # ------------------------------------------------------------------
    # Time-series split
    # ------------------------------------------------------------------

    def time_series_split(self, X, y, n_splits=5):
        """
        TimeSeriesSplit — yields (train_idx, val_idx) tuples.
        Never leaks future data.
        """
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return list(tscv.split(X, y))