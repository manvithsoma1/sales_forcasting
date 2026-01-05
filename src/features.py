import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        self.window_size = 30
        
    def create_features(self, df):
        df = df.copy()
        
        # Ensure 'date' is datetime
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
             df['date'] = pd.to_datetime(df['date'])

        # --- DEFINING COLUMNS ---
        # CRITICAL: 'sales' MUST be here because the scaler expects it.
        cols_to_keep = [
            'sales',          # <--- VITAL: DO NOT REMOVE
            'onpromotion',
            'dcoilwtico',
            'is_holiday',
            'transactions',
            'is_weekend',
            'is_payday'
        ]

        # --- SAFETY PATCH: Fill missing columns with 0 ---
        # This prevents the app from crashing if helper files are imperfect
        for col in cols_to_keep:
            if col not in df.columns:
                # If 'sales' is missing, we are in trouble, but let's try to handle it gracefully
                if col == 'sales':
                    print("⚠️ CRITICAL WARNING: 'sales' column missing! Filling with 0 (Prediction will be flat).")
                else:
                    print(f"⚠️ Warning: '{col}' missing. Filling with 0.")
                
                df[col] = 0
        # -------------------------------------------------

        # Select only the columns we need
        df_featured = df[cols_to_keep].copy()
        
        print(f"Features created. Columns: {df_featured.columns.tolist()}")
        return df_featured