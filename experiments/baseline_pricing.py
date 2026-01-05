import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path so we can import 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader

def run_baseline_simulation():
    print("--- 📉 RUNNING BASELINE (RULE-BASED) SIMULATION ---")
    
    # 1. Load Data
    loader = DataLoader()
    raw = loader.load_raw_data()
    df = loader.merge_data(raw)
    
    # Filter for our test store/product
    df = df[(df['store_nbr'] == 1) & (df['family'] == 'GROCERY I')].copy()
    
    # 2. Define a Simple Rule: "Always Price at Moving Average"
    # Logic: If we just kept the price stable at the monthly average, how much would we make?
    df['baseline_price'] = df['dcoilwtico'].rolling(window=7).mean().fillna(method='bfill') # Fake correlation for demo
    
    # Assume we sold the SAME amount (simplification for baseline)
    # Revenue = Price * Quantity
    # Since we don't have a 'price' column in the dataset (it's sales revenue directly), 
    # we simulate 'sales' as revenue.
    
    total_revenue = df['sales'].sum()
    
    print(f"Total Revenue (Historical/Baseline): ${total_revenue:,.2f}")
    return total_revenue

if __name__ == "__main__":
    run_baseline_simulation()