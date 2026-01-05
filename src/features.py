```python
import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        pass

    def create_features(self, df):
        df = df.copy()
        
        # Ensure 'date' is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            raise ValueError("DataFrame must have a 'date' column.")

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
            'sales',      # Target (keep for modeling)
            'onpromotion',
            'dcoilwtico',
            'is_holiday',
            'transactions',
            'is_weekend',
            'is_payday',
            'day_of_week',
            'month',
            'day_of_year'  # Optional: include if useful for seasonality
        ]

        # --- SAFE MODE FIX: Only keep columns that actually exist ---
        # Calculate "actual" columns present in the dataframe
        valid_cols = [c for c in cols_to_keep if c in df.columns]
        
        # If important columns are missing (Cloud Demo Mode), add them as 0
        missing_cols = set(cols_to_keep) - set(df.columns)
        if missing_cols:
            print(f"⚠️ Warning: Missing columns {missing_cols}. Filling with 0.")
            for c in missing_cols:
                if c not in ['date', 'sales']:  # Don't overwrite these if they exist
                    df[c] = 0  # Fill missing columns with 0 to prevent crash
        
        # Update cols_to_keep to only include now-existing columns
        cols_to_keep = [c for c in cols_to_keep if c in df.columns]
        
        df_featured = df[cols_to_keep].copy()
        # -----------------------------------------------------------

        print(f"Features created. Columns: {df_featured.columns.tolist()}")
        return df_featured
```