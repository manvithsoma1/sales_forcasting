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

        # --- CRITICAL SECTION: COLUMNS TO KEEP ---
        # 'sales' MUST be the first item. Do not remove it.
        cols_to_keep = [
            'sales',         # <--- TARGET VARIABLE (VITAL)
            'onpromotion',
            'dcoilwtico',
            'is_holiday',
            'transactions',
            'is_weekend',
            'is_payday'
        ]

        # --- SAFETY CHECK: Fill missing columns with 0 ---
        for col in cols_to_keep:
            if col not in df.columns:
                # If sales is missing, we must warn loudly
                if col == 'sales':
                    print("⚠️ CRITICAL: 'sales' column missing in input! Filling with 0.")
                else:
                    print(f"⚠️ Warning: '{col}' missing. Filling with 0.")
                df[col] = 0
        # -------------------------------------------------

        # Select columns
        df_featured = df[cols_to_keep].copy()
        
        print(f"Features created. Columns: {df_featured.columns.tolist()}")
        return df_featured