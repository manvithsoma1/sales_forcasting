import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# Custom modules
from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.preprocessing import Preprocessor
from src.weather_service import get_current_weather # <--- NEW IMPORT

# --- CONFIG ---
st.set_page_config(page_title="Dynamic Pricing AI", layout="wide")

# --- API KEY INPUT (For Security) ---
# In a real app, use st.secrets. For now, asking user or hardcode it.
if "weather_api_key" not in st.session_state:
    st.session_state["weather_api_key"] = ""

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    try:
        model = load_model("models/lstm_grocery_v1.h5", compile=False)
        scaler = joblib.load("models/scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None, None
# --------------------------------------------------
# LOAD & FILTER DATA (CACHED)
# --------------------------------------------------
@st.cache_data
def load_data():
    # 1. Try loading the full pipeline (Local Machine)
    if os.path.exists("data/raw/train.csv"):
        loader = DataLoader()
        raw = loader.load_raw_data()
        df = loader.merge_data(raw)
        
        # Filter
        df = df[
            (df["store_nbr"] == 1) &
            (df["family"] == "GROCERY I")
        ].sort_values("date")
        
        return df.tail(1000)
    
    # 2. Fallback to Demo Data (GitHub/Cloud)
    elif os.path.exists("demo_data.csv"):
        print("⚠️ Big data not found. Using Demo Data.")
        df = pd.read_csv("demo_data.csv")
        df['date'] = pd.to_datetime(df['date']) # Ensure date format
        return df.tail(1000)
        
    else:
        st.error("❌ No data found! Please upload 'demo_data.csv' to GitHub.")
        st.stop()
model, scaler = load_resources()
df = load_data()

if model is None: st.stop()

# --- HEADER ---
st.title("🤖 AI Dynamic Pricing Engine")
st.markdown("### Store Sales Forecasting & Promotion Optimization")

# --- SIDEBAR & WEATHER ---
st.sidebar.header("Control Panel")
st.sidebar.info(f"Simulation Date: {df['date'].max().date()}")

st.sidebar.divider()
st.sidebar.subheader("☁️ Real-Time Context")
api_input = st.sidebar.text_input("OpenWeatherMap API Key", type="password")

weather_data = {"condition": "Unknown", "temp": 20}
if api_input:
    weather_data = get_current_weather(api_input)
    st.sidebar.metric("Quito Weather", f"{weather_data['temp']}°C", weather_data['condition'])
    
    # Weather Logic Explanation
    if "Rain" in weather_data['condition']:
        st.sidebar.success("🌧️ RAIN DETECTED: Demand expected to RISE (Home cooking).")
    elif "Clear" in weather_data['condition']:
        st.sidebar.warning("☀️ SUN DETECTED: Demand expected to DROP (Eating out).")
else:
    st.sidebar.warning("Enter API Key to enable Weather Logic")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📈 Forecast", "🧠 Optimization Engine", "💾 Data"])

with tab1:
    st.subheader("Sales Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.tail(300)["date"], y=df.tail(300)["sales"], mode="lines", name="Actual", line=dict(color="#00CC96")))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Hybrid AI Decision Engine")
    st.markdown("""
    This engine combines **Deep Learning (LSTM)** with **Real-Time Rules**.
    1. **LSTM:** Predicts baseline sales based on history.
    2. **Rule Engine:** Adjusts prediction based on live weather (e.g., Rain = +10% Demand).
    """)

    if st.button("🚀 Run Hybrid Simulation", type="primary"):
        with st.spinner("Fetching weather & Running Neural Network..."):
            
            # 1. BASE PREDICTION (LSTM)
            eng = FeatureEngineer()
            df_feat = eng.create_features(df)
            
            pre = Preprocessor()
            pre.scaler = scaler
            
            recent = df_feat.tail(30).copy()
            cols = ["sales"] + [c for c in recent.columns if c not in ["sales", "date"]]
            recent = recent[cols]
            
            scaled_window = scaler.transform(recent)
            input_seq = scaled_window.reshape(1, 30, -1)
            
            # Scenario 1: No Promo
            seq_no = input_seq.copy()
            seq_no[0, -1, 2] = 0
            pred_no_raw = model.predict(seq_no, verbose=0)
            
            # Scenario 2: Promo
            seq_yes = input_seq.copy()
            seq_yes[0, -1, 2] = 1
            pred_yes_raw = model.predict(seq_yes, verbose=0)
            
            # Inverse Scale
            dummy = np.zeros((1, scaled_window.shape[1]))
            dummy[0, 0] = pred_no_raw[0, 0]
            base_sales_no = scaler.inverse_transform(dummy)[0, 0]
            
            dummy[0, 0] = pred_yes_raw[0, 0]
            base_sales_yes = scaler.inverse_transform(dummy)[0, 0]
            
            # 2. WEATHER ADJUSTMENT (The Twist)
            modifier = 1.0
            reason = "Normal Weather"
            
            condition = weather_data['condition']
            if "Rain" in condition or "Drizzle" in condition or "Thunderstorm" in condition:
                modifier = 1.10 # +10% Demand
                reason = "Rainy (High Demand for Grocery)"
            elif "Clear" in condition or "Sun" in condition:
                modifier = 0.95 # -5% Demand
                reason = "Sunny (Low Demand / Eating Out)"
            
            final_sales_no = base_sales_no * modifier
            final_sales_yes = base_sales_yes * modifier
            
            # 3. PROFIT CALC
            profit_no = (10 - 6) * final_sales_no
            profit_yes = (8 - 6) * final_sales_yes
            
            # 4. DISPLAY
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ☁️ Weather Impact")
                st.metric("Condition", condition)
                st.metric("AI Adjustment", f"{modifier}x", help="Multiplier applied to LSTM prediction")
                st.caption(f"Reasoning: {reason}")
            
            with col2:
                st.markdown("### 💰 Financial Outcome")
                st.metric("Sales (No Promo)", f"{final_sales_no:.0f}", delta=f"{final_sales_no - base_sales_no:.0f} (Weather)")
                st.metric("Profit (No Promo)", f"${profit_no:.2f}")
                
                st.divider()
                
                st.metric("Sales (With Promo)", f"{final_sales_yes:.0f}")
                st.metric("Profit (With Promo)", f"${profit_yes:.2f}", delta=f"${profit_yes - profit_no:.2f}")

            if profit_yes > profit_no:
                st.success(f"✅ RECOMMENDATION: RUN PROMOTION! (Weather-Adjusted Profit: +${profit_yes - profit_no:.2f})")
            else:
                st.error("❌ RECOMMENDATION: DO NOT PROMOTE.")

with tab3:
    st.dataframe(df)