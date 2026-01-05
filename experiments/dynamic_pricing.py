import sys
import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.preprocessing import Preprocessor

def run_ai_simulation():
    print("--- 🤖 RUNNING AI DYNAMIC PRICING SIMULATION ---")
    
    # 1. Load Model & Scaler
    try:
        model = load_model('models/lstm_grocery_v1.h5')
        scaler = joblib.load('models/scaler.pkl')
        print("✅ Model and Scaler loaded.")
    except:
        print("❌ Error: Model not found. Run main.py first.")
        return

    # 2. Load Recent Data (Last 30 days) to predict "Tomorrow"
    loader = DataLoader()
    raw = loader.load_raw_data()
    df = loader.merge_data(raw)
    df = df[(df['store_nbr'] == 1) & (df['family'] == 'GROCERY I')].sort_values('date')
    
    # Process
    eng = FeatureEngineer()
    df_feat = eng.create_features(df)
    
    pre = Preprocessor()
    # We need to scale it using the LOADED scaler, not a new one
    pre.scaler = scaler 
    
    # Prepare last 30 days
    last_30_days = df_feat.tail(30).copy()
    
    # Need to match column order exactly
    cols = ['sales'] + [c for c in last_30_days.columns if c != 'sales' and c != 'date']
    last_30_days = last_30_days[cols]
    
    # Transform
    scaled_data = scaler.transform(last_30_days)
    
    # Reshape for LSTM (1 sample, 30 timesteps, 7 features)
    input_seq = scaled_data.reshape(1, 30, -1)
    
    # 3. Predict Revenue for Tomorrow
    predicted_scaled = model.predict(input_seq, verbose=0)
    
    # Inverse transform to get real money value
    dummy = np.zeros((1, 7))
    dummy[0, 0] = predicted_scaled[0, 0]
    predicted_sales = scaler.inverse_transform(dummy)[0, 0]
    
    print(f"🔮 AI Predicted Sales Revenue for Tomorrow: ${predicted_sales:.2f}")

if __name__ == "__main__":
    run_ai_simulation()