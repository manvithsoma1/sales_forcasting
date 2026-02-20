import pandas as pd
import numpy as np
import joblib
from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.preprocessing import Preprocessor
from src.model import LSTMModel
from src.evaluation import Evaluator
from src.optimization import PromotionOptimizer  # <--- Make sure this import is here

def main():
    # 1. Load Data
    loader = DataLoader()
    raw_data = loader.load_raw_data()
    df_merged = loader.merge_data(raw_data)
    
    # Filter for Store 1, Grocery I
    print("\n[INFO] Filtering for Store 1, Product Family 'GROCERY I'...")
    df_subset = df_merged[
        (df_merged['store_nbr'] == 1) & 
        (df_merged['family'] == 'GROCERY I')
    ].sort_values('date')
    
    # 2. Features
    engineer = FeatureEngineer()
    df_featured = engineer.create_features(df_subset)
    
    # 3. Preprocessing
    preprocessor = Preprocessor()
    cols = ['sales'] + [c for c in df_featured.columns if c != 'sales' and c != 'date']
    df_ordered = df_featured[cols]
    
    # Scale & SAVE THE SCALER
    df_scaled, scaler = preprocessor.scale_data(df_ordered)
    joblib.dump(scaler, 'models/scaler.pkl') 
    
    X, y = preprocessor.create_sequences(df_scaled.values)
    
    # 4. Train
    input_shape = (X.shape[1], X.shape[2]) 
    lstm = LSTMModel(input_shape=input_shape)
    
    history = lstm.train(X, y)  # Uses config epochs + early stopping
    
    # 4b. Evaluate
    evaluator = Evaluator()
    evaluator.plot_loss(history)
    y_pred = lstm.model.predict(X, verbose=0).flatten()
    evaluator.calculate_metrics(y, y_pred)
    evaluator.plot_predictions(y, y_pred, start_idx=0, length=min(100, len(y)))
    
    lstm.save_model('models/lstm_grocery_v1.h5')
    
    # --- 5. OPTIMIZATION SIMULATION (The Missing Piece) ---
    print("\n[INFO] Starting Promotion Optimization Logic...")
    
    # Take the VERY LAST 30 days of data to simulate "Tomorrow"
    recent_data_window = df_scaled.values[-30:] 
    
    # Run the optimizer
    optimizer = PromotionOptimizer(lstm.model, scaler)
    optimizer.optimize(recent_data_window)

if __name__ == "__main__":
    main()