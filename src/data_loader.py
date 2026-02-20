import pandas as pd
import os
import yaml # Reads config.yaml

class DataLoader:
    def __init__(self, config_path='config/config.yaml'): #Runs automatically when the class is created
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.raw_path = self.config['data']['raw_path']
        self.files = self.config['data']['files']
    
    
    def load_raw_data(self):
        """Loads all raw CSVs into a dictionary of DataFrames."""
        print("Loading raw data...")
        data = {}
        # Load Train (support .csv or .zip)
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
        # Load Stores
        data['stores'] = pd.read_csv(os.path.join(self.raw_path, self.files['stores']))
        # Load Oil
        data['oil'] = pd.read_csv(os.path.join(self.raw_path, self.files['oil']))
        # Load Holidays
        data['holidays'] = pd.read_csv(os.path.join(self.raw_path, self.files['holidays']))
        # Load Transactions (store-level daily transaction count)
        if 'transactions' in self.files and os.path.exists(os.path.join(self.raw_path, self.files['transactions'])):
            data['transactions'] = pd.read_csv(os.path.join(self.raw_path, self.files['transactions']))
        
        print("Raw data loaded successfully.")
        return data  
    def merge_data(self, data):
        """
        Merges the disparate datasets (Oil, Stores, Holidays) into the main Train set.
        """
        print("Merging data...")
        df_train = data['train']
        
        # 1. Merge with Stores (adds city, state, type, cluster)
        df_merged = df_train.merge(data['stores'], on='store_nbr', how='left')
        
        # 2. Merge with Oil (Economic Indicator)
        # Note: Oil data has dates, so we merge on date.
        data['oil']['date'] = pd.to_datetime(data['oil']['date'])
        df_merged['date'] = pd.to_datetime(df_merged['date'])
        
        # Forward fill missing oil values (if oil price is missing for weekends, use Friday's price)
        # We do this before merging for cleaner data
        oil_df = data['oil'].sort_values('date').set_index('date').resample('D').ffill().reset_index()
        
        df_merged = df_merged.merge(oil_df, on='date', how='left')
        
        # 3. Simple Holiday Merge (Binary Flag for now)
        # We just want to know "Is this date a holiday?"
# New code (adds .copy())
        holidays = data['holidays'][['date', 'type', 'transferred']].copy()
        holidays['date'] = pd.to_datetime(holidays['date'])
        
        # Filter out transferred holidays (they aren't holidays on that day)
        holidays = holidays[holidays['transferred'] == False]
        
        # Create a simpler 'is_holiday' column
        holidays['is_holiday'] = 1
        holidays = holidays[['date', 'is_holiday']].drop_duplicates(subset='date')
        
        df_merged = df_merged.merge(holidays, on='date', how='left')
        
        # Fill NaN for is_holiday with 0
        df_merged['is_holiday'] = df_merged['is_holiday'].fillna(0)
        
        # 4. Merge Transactions (daily store transaction count - demand signal)
        if 'transactions' in data:
            trans = data['transactions'].copy()
            trans['date'] = pd.to_datetime(trans['date'])
            df_merged = df_merged.merge(trans[['date', 'store_nbr', 'transactions']], on=['date', 'store_nbr'], how='left')
            df_merged['transactions'] = df_merged['transactions'].fillna(0)
        
        print(f"Data Merged. Shape: {df_merged.shape}")
        return df_merged

if __name__ == "__main__":
    # Test the loader
    loader = DataLoader()
    raw_data = loader.load_raw_data()
    df_final = loader.merge_data(raw_data)
    print(df_final.head())