import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        pass

    def create_features(self, df):
        """
        Extracts time-based and business-relevant features.
        """
        print("Engineering features...")

        # Always work on a copy (prevents SettingWithCopyWarning)
        df = df.copy()

        # Ensure datetime
        df['date'] = pd.to_datetime(df['date'])

        # --------------------
        # Time-based features
        # --------------------
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear

        # Weekend indicator (vectorized)
        df['is_weekend'] = np.where(df['day_of_week'] >= 5, 1, 0)

        # Payday indicator
        # Payday = 15th OR last day of month
        df['is_payday'] = np.where(
            (df['date'].dt.day == 15) | (df['date'].dt.is_month_end),
            1,
            0
        )

        # --------------------
        # Final feature set
        # --------------------
        cols_to_keep = [
            'date',
            'sales',
            'onpromotion',
            'dcoilwtico',
            'is_holiday',
            'day_of_week',
            'month',
            'is_payday'
        ]

       # --- SAFE MODE FIX: Only keep columns that actually exist ---
        # calculate "actual" columns present in the dataframe
        valid_cols = [c for c in cols_to_keep if c in df.columns]
        
        # If important columns are missing (Cloud Demo Mode), add them as 0
        missing_cols = set(cols_to_keep) - set(valid_cols)
        if missing_cols:
            print(f"⚠️ Warning: Missing columns {missing_cols}. Filling with 0.")
            for c in missing_cols:
                df[c] = 0  # Fill missing columns with 0 to prevent crash
        
        df_featured = df[cols_to_keep].copy()
        # -----------------------------------------------------------

        print(f"Features created. Columns: {df_featured.columns.tolist()}")
        return df_featured
