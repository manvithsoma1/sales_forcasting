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

        # Define the columns we expect
        # CRITICAL FIX: Added 'sales' back to this list!
        cols_to_keep = [
            'sales',          # <--- THIS WAS MISSING
            'onpromotion',
            'dcoilwtico',
            'is_holiday',
            'transactions',
            'is_weekend',
            'is_payday'
        ]

        # --- SAFETY PATCH: Create missing columns with 0 instead of crashing ---
        for col in cols_to_keep:
            if col not in df.columns:
                print(f"⚠️ Warning: '{col}' missing in features.py. Filling with 0.")
                df[col] = 0
        # -----------------------------------------------------------------------

        # Now it is safe to select them
        df_featured = df[cols_to_keep].copy()
        
        print(f"Features created. Columns: {df_featured.columns.tolist()}")
        return df_featured