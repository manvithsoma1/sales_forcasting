import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yaml

class Preprocessor:
    def __init__(self, config_path='config/config.yaml'):
        # Load config safely
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.look_back = self.config['model']['look_back_days']

    def scale_data(self, df):
        """
        Scales numeric features safely.
        """
        print("Scaling data...")
        
        # 1. Identify numeric columns only (ignores 'date' if it exists)
        df_numeric = df.select_dtypes(include=[np.number])
        
        # FIX: Fill any remaining NaNs with 0 before scaling
        if df_numeric.isna().any().any():
            print("[WARNING] Found NaNs in data. Filling with 0.")
            df_numeric = df_numeric.fillna(0)

        # 2. Ensure 'sales' is the first column (Critical for our sequence logic)
        if 'sales' not in df_numeric.columns:
             raise ValueError("Expected 'sales' column not found in input data.")
             
        cols = ['sales'] + [c for c in df_numeric.columns if c != 'sales']
        df_numeric = df_numeric[cols]
        
        # 3. Scale
        scaled_data = self.scaler.fit_transform(df_numeric)
        
        df_scaled = pd.DataFrame(
            scaled_data, 
            columns=df_numeric.columns
        )
        
        return df_scaled, self.scaler

    def create_sequences(self, data):
        """
        Converts data into 3D sequences for LSTM (Samples, TimeSteps, Features).
        """
        print("Creating sequences for LSTM...")
        
        X, y = [], []
        
        # Loop through data to create sequences
        # Input: Past 'look_back' days
        # Output: Sales on the next day
        for i in range(len(data) - self.look_back):
            X.append(data[i : i + self.look_back])
            y.append(data[i + self.look_back, 0]) # 0 is the index of 'sales' column
            
        return np.array(X), np.array(y)