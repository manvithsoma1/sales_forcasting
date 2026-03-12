"""
Data Loader Module — v2.0
Loads and merges Corporación Favorita datasets with multi-store/family support.
"""

import pandas as pd
import os
import yaml


class DataLoader:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.raw_path = self.config['data']['raw_path']
        self.files = self.config['data']['files']

    # ------------------------------------------------------------------
    # Load raw CSV files
    # ------------------------------------------------------------------

    def load_raw_data(self):
        """Loads all raw CSVs into a dictionary of DataFrames."""
        print("Loading raw data...")
        data = {}

        # Train (support .csv or .zip)
        train_path = os.path.join(self.raw_path, self.files['train'])
        zip_path = os.path.join(self.raw_path, 'train.zip')
        if os.path.exists(train_path):
            data['train'] = pd.read_csv(train_path)
        elif os.path.exists(zip_path):
            data['train'] = pd.read_csv(zip_path, compression='zip')
        else:
            raise FileNotFoundError(f"Train data not found at {train_path} or {zip_path}")

        if 'unit_sales' in data['train'].columns and 'sales' not in data['train'].columns:
            data['train'].rename(columns={'unit_sales': 'sales'}, inplace=True)

        # Helper files
        data['stores'] = pd.read_csv(os.path.join(self.raw_path, self.files['stores']))
        data['oil'] = pd.read_csv(os.path.join(self.raw_path, self.files['oil']))
        data['holidays'] = pd.read_csv(os.path.join(self.raw_path, self.files['holidays']))

        if 'transactions' in self.files:
            trans_path = os.path.join(self.raw_path, self.files['transactions'])
            if os.path.exists(trans_path):
                data['transactions'] = pd.read_csv(trans_path)

        print("Raw data loaded successfully.")
        return data

    # ------------------------------------------------------------------
    # Merge datasets
    # ------------------------------------------------------------------

    def merge_data(self, data):
        """
        Merges Oil, Stores, Holidays, Transactions into Train set.
        Returns full merged DataFrame (all stores/families).
        """
        print("Merging data...")
        df = data['train'].copy()
        df['date'] = pd.to_datetime(df['date'])

        # 1. Stores — adds city, state, type, cluster
        df = df.merge(data['stores'], on='store_nbr', how='left')

        # 2. Oil — economic indicator (forward-filled for weekends)
        oil = data['oil'].copy()
        oil['date'] = pd.to_datetime(oil['date'])
        oil = oil.sort_values('date').set_index('date').resample('D').ffill().reset_index()
        df = df.merge(oil, on='date', how='left')

        # 3. Holidays — preserve locale for national/regional distinction
        holidays = data['holidays'].copy()
        holidays['date'] = pd.to_datetime(holidays['date'])
        holidays_filtered = holidays[holidays['transferred'] == False].copy()
        holiday_flag = holidays_filtered[['date']].drop_duplicates()
        holiday_flag['is_holiday'] = 1
        df = df.merge(holiday_flag, on='date', how='left')
        df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)

        # 4. Transactions
        if 'transactions' in data:
            trans = data['transactions'].copy()
            trans['date'] = pd.to_datetime(trans['date'])
            df = df.merge(
                trans[['date', 'store_nbr', 'transactions']],
                on=['date', 'store_nbr'], how='left'
            )
            df['transactions'] = df['transactions'].fillna(0)

        # Fill remaining NaNs
        df['dcoilwtico'] = df['dcoilwtico'].ffill().bfill().fillna(0)
        df['onpromotion'] = df['onpromotion'].fillna(0).astype(int)

        print(f"Data merged. Shape: {df.shape}")
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_holidays_raw(self):
        """Returns raw holidays DataFrame for detailed feature engineering."""
        path = os.path.join(self.raw_path, self.files['holidays'])
        return pd.read_csv(path)

    def get_store_families(self, df):
        """Returns unique store numbers and product families for dashboard selectors."""
        stores = sorted(df['store_nbr'].unique().tolist()) if 'store_nbr' in df.columns else [1]
        families = sorted(df['family'].unique().tolist()) if 'family' in df.columns else ['GROCERY I']
        return stores, families

    def filter_subset(self, df, store_nbr=None, family=None):
        """Filter merged data to specific store/family combo."""
        result = df.copy()
        if store_nbr is not None:
            result = result[result['store_nbr'] == store_nbr]
        if family is not None:
            result = result[result['family'] == family]
        return result.sort_values('date').reset_index(drop=True)


if __name__ == "__main__":
    loader = DataLoader()
    raw_data = loader.load_raw_data()
    df_final = loader.merge_data(raw_data)
    stores, families = loader.get_store_families(df_final)
    print(f"Stores: {len(stores)}, Families: {len(families)}")
    print(df_final.head())