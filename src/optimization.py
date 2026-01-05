import numpy as np
import pandas as pd

class PromotionOptimizer:
    def __init__(self, model, scaler, look_back=30):
        self.model = model
        self.scaler = scaler
        self.look_back = look_back

    def optimize(self, recent_data):
        """
        Simulates two scenarios (Promo vs No Promo) and calculates profit.
        """
        print("\n--- 🤖 RUNNING PROMOTION SIMULATION ---")
        
        # 1. Prepare Base Input (The last 30 days of data)
        # We need the data to be scaled already
        current_seq = recent_data[-self.look_back:]
        
        # We need to reshape it to (1, 30, 7) for the LSTM
        input_seq = current_seq.reshape(1, self.look_back, -1)
        
        # --- SCENARIO 1: NO PROMOTION ---
        # "onpromotion" is the column at index 2 (based on your features list)
        # Let's verify columns: ['sales', 'onpromotion', 'dcoilwtico', 'is_holiday'...]
        PROMO_IDX = 2 
        
        seq_no_promo = input_seq.copy()
        # Set the most recent day's promo status to 0 (No Promo)
        seq_no_promo[0, -1, PROMO_IDX] = 0 
        
        pred_no_promo_scaled = self.model.predict(seq_no_promo, verbose=0)
        
        # --- SCENARIO 2: ACTIVATE PROMOTION ---
        seq_promo = input_seq.copy()
        # Set the most recent day's promo status to 1 (Yes Promo)
        seq_promo[0, -1, PROMO_IDX] = 1
        
        pred_promo_scaled = self.model.predict(seq_promo, verbose=0)
        
        # --- CALCULATE PROFIT ---
        # We need to un-scale the predictions to get real units
        sales_no_promo = self._inverse_transform_sales(pred_no_promo_scaled, current_seq)
        sales_promo = self._inverse_transform_sales(pred_promo_scaled, current_seq)
        
        # Business Logic Assumptions (You can explain this in your interview)
        # Item Price = $10. Cost = $6.
        # Promotion Discount = 20% off (Price becomes $8)
        base_price = 10
        cost = 6
        
        # Profit = (Price - Cost) * Quantity
        profit_no_promo = (base_price - cost) * sales_no_promo
        profit_promo = ((base_price * 0.8) - cost) * sales_promo
        
        # --- REPORT ---
        print(f"🔮 Prediction (No Promo): {sales_no_promo:.0f} units -> Profit: ${profit_no_promo:.2f}")
        print(f"🔥 Prediction (With Promo): {sales_promo:.0f} units -> Profit: ${profit_promo:.2f}")
        
        if profit_promo > profit_no_promo:
            lift = profit_promo - profit_no_promo
            print(f"✅ RECOMMENDATION: RUN THE PROMOTION! (Profit Increase: +${lift:.2f})")
        else:
            print(f"❌ RECOMMENDATION: DO NOT PROMOTE. (You would lose money).")

    def _inverse_transform_sales(self, pred_value, reference_data):
        """
        Helper to un-scale just the sales number.
        """
        # Create a dummy row with the same shape as original data
        dummy_row = np.zeros((1, reference_data.shape[1]))
        # Put our predicted sales in the first column
        dummy_row[0, 0] = pred_value
        # Inverse transform
        return self.scaler.inverse_transform(dummy_row)[0, 0]