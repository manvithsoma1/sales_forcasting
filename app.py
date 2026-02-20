import os
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
from src.weather_service import get_current_weather

# --- CONFIG ---
st.set_page_config(
    page_title="AI Dynamic Pricing Engine",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLING & ANIMATIONS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Base theme */
    .stApp { 
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        font-family: 'Inter', -apple-system, sans-serif !important;
    }
    .main .block-container { padding-top: 1.5rem; max-width: 1400px; }
    
    /* Hero section animation */
    .hero-badge {
        display: inline-block;
        background: linear-gradient(90deg, #06b6d4, #3b82f6);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        animation: fadeInDown 0.6s ease-out;
    }
    h1 {
        color: #f8fafc !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
        animation: fadeInUp 0.7s ease-out 0.1s both;
    }
    h2, h3 { color: #e2e8f0 !important; animation: fadeInUp 0.6s ease-out both; }
    
    /* Tab animation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(30, 41, 59, 0.6);
        padding: 6px;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #06b6d4, #3b82f6) !important;
    }
    
    /* Metric cards with subtle animation */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
        padding: 1.25rem !important;
        border-radius: 12px !important;
        border: 1px solid rgba(148, 163, 184, 0.2);
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeInScale 0.5s ease-out both;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(6, 182, 212, 0.15);
    }
    
    /* Chart container */
    [data-testid="stPlotlyChart"] {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid rgba(148, 163, 184, 0.15);
        animation: fadeIn 0.8s ease-out;
    }
    
    /* Recommendation box */
    .rec-box {
        padding: 1.5rem 2rem;
        border-radius: 16px;
        font-size: 1.15rem;
        font-weight: 600;
        text-align: center;
        animation: pulseGlow 2s ease-in-out infinite;
        border: 1px solid;
    }
    .rec-success {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(22, 163, 74, 0.15));
        border-color: rgba(34, 197, 94, 0.5);
        color: #86efac;
    }
    .rec-error {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(185, 28, 28, 0.15));
        border-color: rgba(239, 68, 68, 0.5);
        color: #fca5a5;
    }
    
    /* Keyframe animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInScale {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    [data-testid="stMetric"]:nth-child(1) { animation-delay: 0.1s; }
    [data-testid="stMetric"]:nth-child(2) { animation-delay: 0.2s; }
    [data-testid="stMetric"]:nth-child(3) { animation-delay: 0.3s; }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 20px rgba(6, 182, 212, 0.2); }
        50% { box-shadow: 0 0 40px rgba(6, 182, 212, 0.4); }
    }
    
    /* Sidebar polish */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
    }
    [data-testid="stSidebar"] .stMarkdown { color: #cbd5e1 !important; }
    
    /* Button hover */
    .stButton > button {
        background: linear-gradient(90deg, #06b6d4, #3b82f6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px rgba(6, 182, 212, 0.4) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.5) !important;
    }
    
    /* Caption & text */
    .stCaptionContainer, p, .stMarkdown { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

# --- API KEY INPUT (For Security) ---
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
# LOAD & FILTER DATA (REAL ZIPPED DATA)
# --------------------------------------------------
@st.cache_data
def load_data():
    # Define paths
    zip_path = "data/raw/train.zip"
    oil_path = "data/raw/oil.csv"
    stores_path = "data/raw/stores.csv"
    trans_path = "data/raw/transactions.csv"
    holidays_path = "data/raw/holidays_events.csv"

    # Check if the Zip file exists
    if os.path.exists(zip_path):
        # print("📦 Loading REAL data from ZIP...")
        
        # 1. READ DIRECTLY FROM ZIP (Pandas can do this!)
        raw = pd.read_csv(zip_path, compression='zip')
        raw.rename(columns={"unit_sales": "sales"}, inplace=True)
        # 2. LOAD REAL HELPER FILES
        try:
            oil = pd.read_csv(oil_path)
            stores = pd.read_csv(stores_path)
            trans = pd.read_csv(trans_path)
            holidays = pd.read_csv(holidays_path)
        except FileNotFoundError as e:
            st.error(f"❌ Missing helper file: {e}. Please check GitHub.")
            st.stop()

        # 3. MERGE DATA (Recreating the original pipeline)
        # Convert dates
        raw['date'] = pd.to_datetime(raw['date'])
        oil['date'] = pd.to_datetime(oil['date'])
        holidays['date'] = pd.to_datetime(holidays['date'])
        
        # Merge Oil
        df = pd.merge(raw, oil, on='date', how='left')
        
        # Merge Stores
        df = pd.merge(df, stores, on='store_nbr', how='left')
        
        # Merge Holidays (Simplified Logic)
        holidays = holidays[~holidays['transferred']]
        holidays = holidays[['date', 'locale', 'description']]
        holidays = holidays.drop_duplicates(subset=['date']) 
        df = pd.merge(df, holidays, on='date', how='left')
        
        # Create the 'is_holiday' flag
        df['is_holiday'] = df['locale'].notnull().astype(int)

        # Fill Oil Missing Values (Interpolation)
        df['dcoilwtico'] = df['dcoilwtico'].ffill().bfill()

        # Merge Transactions (store-level demand signal)
        trans['date'] = pd.to_datetime(trans['date'])
        df = df.merge(trans[['date', 'store_nbr', 'transactions']], on=['date', 'store_nbr'], how='left')
        df['transactions'] = df['transactions'].fillna(0)

        # 4. FILTER (Crucial for Cloud Memory!)
        # Filter for Store 1, Grocery I
        df_filtered = df[
            (df["store_nbr"] == 1) &
            (df["family"] == "GROCERY I")
        ].sort_values("date")
        
        return df_filtered.tail(2000)
        
    else:
        st.error(f"❌ Could not find {zip_path}. Please upload 'train.zip' and helper CSVs to 'data/raw/'.")
        st.stop()

# --- MAIN APP LOGIC ---
model, scaler = load_resources()
df = load_data()

if model is None: st.stop()

# --- HEADER ---
st.markdown('<p class="hero-badge">AI-POWERED DECISION ENGINE</p>', unsafe_allow_html=True)
st.title("🛒 Dynamic Pricing & Promotion Optimizer")
st.markdown("**Forecast sales, compare scenarios, and get AI recommendations** — LSTM predictions + real-time weather for smarter promotion decisions.")
st.caption("📍 Store 1 · Grocery I · Quito, Ecuador")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    st.caption(f"Last data date: {df['date'].max().date()}")
    st.divider()
    st.subheader("☁️ Real-Time Weather")
    st.caption("Enter your [OpenWeatherMap](https://openweathermap.org/api) key for weather-adjusted recommendations.")
    api_input = st.text_input("API Key", type="password", placeholder="Your API key...", label_visibility="collapsed")
    
    weather_data = {"condition": "Unknown", "temp": 20}
    if api_input:
        weather_data = get_current_weather(api_input)
        st.metric("Quito", f"{weather_data['temp']}°C", weather_data['condition'])
        if "Rain" in weather_data['condition']:
            st.success("🌧️ Higher demand expected (stay home, cook more)")
        elif "Clear" in weather_data['condition']:
            st.warning("☀️ Lower demand expected (eating out)")
    else:
        st.info("Add API key to enable weather logic. Simulation will use default.")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📈 Sales Forecast", "🧠 Promotion Optimizer", "💾 Raw Data"])

with tab1:
    st.subheader("📈 Historical Sales Trend")
    st.caption("Last 300 days — primary input for the LSTM model")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.tail(300)["date"], y=df.tail(300)["sales"],
        mode="lines", name="Actual Sales",
        line=dict(color="#22d3ee", width=2.5, shape="spline", smoothing=0.3),
        fill='tozeroy', fillcolor='rgba(34, 211, 238, 0.15)'
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8', size=12),
        hovermode="x unified",
        margin=dict(l=50, r=30, t=30, b=50),
        xaxis=dict(showgrid=True, gridcolor='rgba(148,163,184,0.15)', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(148,163,184,0.15)', zeroline=False),
        xaxis_title="Date", yaxis_title="Sales",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)')
    )
    fig.update_xaxes(showline=True, linecolor='rgba(148,163,184,0.3)')
    fig.update_yaxes(showline=True, linecolor='rgba(148,163,184,0.3)')
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})

with tab2:
    st.subheader("🧠 Run Promotion Decision")
    st.markdown("Compare **No Promotion** vs **With Promotion** — LSTM + weather logic + profit math.")
    with st.expander("ℹ️ How it works", expanded=False):
        st.markdown("""
        - **LSTM** predicts sales from last 30 days (oil, holidays, transactions)
        - **Weather** adjusts: Rain +10% demand, Sunny −5%
        - **Economics:** Base $10 → Promo $8 (20% off), Cost $6
        """)
    if st.button("🚀 Run Simulation", type="primary"):
        with st.spinner("🔄 Running AI simulation — fetching weather & LSTM prediction..."):
            
            # 1. BASE PREDICTION (LSTM)
            eng = FeatureEngineer()
            df_feat = eng.create_features(df)
            
            # Must match scaler fit-time columns exactly
            cols = ["sales", "onpromotion", "dcoilwtico", "is_holiday", "day_of_week", "month", "is_payday"]
            if "sales" not in df_feat.columns:
                st.error("❌ 'sales' missing after feature engineering. Check load_data().")
                st.stop()
            
            recent = df_feat[cols].tail(30).copy()
            scaled_window = scaler.transform(recent)
            input_seq = scaled_window.reshape(1, 30, -1)

            # onpromotion is index 1 in feature order
            PROMO_IDX = 1
            
            # Scenario 1: No Promo
            seq_no = input_seq.copy()
            seq_no[0, -1, PROMO_IDX] = 0 
            pred_no_raw = model.predict(seq_no, verbose=0)
            
            # Scenario 2: Promo
            seq_yes = input_seq.copy()
            seq_yes[0, -1, PROMO_IDX] = 1
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
                st.caption(f"_Reasoning: {reason}_")
            
            with col2:
                st.markdown("### 💰 Financial Outcome")
                st.metric("Sales (No Promo)", f"{final_sales_no:,.0f}", delta=f"{final_sales_no - base_sales_no:+,.0f} (weather)")
                st.metric("Profit (No Promo)", f"${profit_no:,.2f}")
                st.divider()
                st.metric("Sales (With Promo)", f"{final_sales_yes:,.0f}")
                st.metric("Profit (With Promo)", f"${profit_yes:,.2f}", delta=f"${profit_yes - profit_no:+,.2f}")

            st.divider()
            delta_profit = profit_yes - profit_no
            rec_class = "rec-success" if profit_yes > profit_no else "rec-error"
            rec_icon = "✅" if profit_yes > profit_no else "❌"
            rec_text = f"**Run promotion** — Extra profit: +${delta_profit:,.2f}" if profit_yes > profit_no else f"**Do not promote** — Would lose ${abs(delta_profit):,.2f}"
            st.markdown(f'<div class="rec-box {rec_class}">{rec_icon} Recommendation: {rec_text}</div>', unsafe_allow_html=True)

with tab3:
    st.subheader("💾 Training Data Sample")
    st.caption(f"Rows: {len(df):,} · Columns: {', '.join(df.columns[:8])}{'...' if len(df.columns) > 8 else ''}")
    display_df = df.style.background_gradient(subset=['sales'], cmap='Blues', axis=0) if 'sales' in df.columns else df
    st.dataframe(display_df, use_container_width=True, height=400)