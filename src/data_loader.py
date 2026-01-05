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
        # Load Train
        data['train'] = pd.read_csv(os.path.join(self.raw_path, self.files['train']))
        # Load Stores
        data['stores'] = pd.read_csv(os.path.join(self.raw_path, self.files['stores']))
        # Load Oil
        data['oil'] = pd.read_csv(os.path.join(self.raw_path, self.files['oil']))
        # Load Holidays
        data['holidays'] = pd.read_csv(os.path.join(self.raw_path, self.files['holidays']))
        
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
        
        print(f"Data Merged. Shape: {df_merged.shape}")
        return df_merged

if __name__ == "__main__":
    # Test the loader
    loader = DataLoader()
    raw_data = loader.load_raw_data()
    df_final = loader.merge_data(raw_data)
    print(df_final.head())