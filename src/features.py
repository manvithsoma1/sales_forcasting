import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        pass

    def create_features(self, df):
        print("Engineering features...")
        df = df.copy()

        # -----------------------------
        # Date is mandatory
        # -----------------------------
        if 'date' not in df.columns:
            raise ValueError("❌ 'date' column is required")

        df['date'] = pd.to_datetime(df['date'])

        # -----------------------------
        # Ensure required base columns
        # -----------------------------
        required_defaults = {
            'sales': 0,          # TARGET (MUST EXIST)
            'onpromotion': 0,
            'dcoilwtico': 0.0,
            'is_holiday': 0,
            'transactions': 0
        }

        for col, default in required_defaults.items():
            if col not in df.columns:
                print(f"⚠️ Warning: '{col}' missing. Filling with {default}.")
                df[col] = default

        # -----------------------------
        # Time-based features (must match scaler fit-time columns)
        # -----------------------------
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['is_payday'] = np.where(
            (df['date'].dt.day == 15) | (df['date'].dt.is_month_end),
            1,
            0
        )

        # -----------------------------
        # FINAL MODEL FEATURES (order must match trained scaler)
        # sales MUST be index 0, onpromotion index 1
        # -----------------------------
        cols_to_keep = [
            'sales',        # index 0 (target)
            'onpromotion',
            'dcoilwtico',
            'is_holiday',
            'day_of_week',
            'month',
            'is_payday'
        ]

        df_featured = df[cols_to_keep].copy()

        print(f"Features created. Columns: {df_featured.columns.tolist()}")
        return df_featured
