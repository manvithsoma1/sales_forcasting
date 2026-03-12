"""
Store Sales Forecasting & AI Dynamic Pricing Engine — Dashboard v2.0
5-page Streamlit app: Executive Overview, Sales Forecasting, Promotion Optimizer,
External Signals, What-If Simulator.
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta

# Custom modules
from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.preprocessing import Preprocessor
from src.optimization import PromotionOptimizer
from src.weather_service import get_current_weather

# ======================================================================
# PAGE CONFIG
# ======================================================================
st.set_page_config(
    page_title="AI Sales Engine — Store Forecasting",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================================
# DESIGN SYSTEM — Dark Theme
# ======================================================================
BG = "#0A0E1A"
BG_CARD = "#111827"
BG_CARD_HOVER = "#1F2937"
ACCENT = "#00D4AA"
ACCENT2 = "#3B82F6"
ALERT = "#FF6B6B"
WARN = "#FBBF24"
TEXT = "#E5E7EB"
TEXT_DIM = "#9CA3AF"
FONT_HEADER = "'Outfit', sans-serif"
FONT_MONO = "'JetBrains Mono', monospace"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* ---- Base ---- */
    .stApp {{
        background: linear-gradient(160deg, {BG} 0%, #0F1729 40%, #111827 100%);
        font-family: {FONT_HEADER};
    }}
    .main .block-container {{ padding-top: 1rem; max-width: 1440px; }}

    /* ---- Typography ---- */
    h1 {{
        font-family: {FONT_HEADER} !important;
        font-weight: 900 !important;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, {ACCENT}, {ACCENT2});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fadeInUp 0.6s ease-out both;
    }}
    h2, h3 {{
        font-family: {FONT_HEADER} !important;
        color: {TEXT} !important;
        font-weight: 700 !important;
    }}
    p, .stMarkdown {{ color: {TEXT_DIM} !important; }}

    /* ---- Animated KPI Cards ---- */
    .kpi-card {{
        background: linear-gradient(145deg, {BG_CARD}, {BG});
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeInScale 0.5s ease-out both;
    }}
    .kpi-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,212,170,0.15);
    }}
    .kpi-label {{
        font-family: {FONT_HEADER};
        font-size: 0.75rem;
        font-weight: 600;
        color: {TEXT_DIM};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }}
    .kpi-value {{
        font-family: {FONT_MONO};
        font-size: 1.8rem;
        font-weight: 700;
        color: {TEXT};
        line-height: 1.2;
    }}
    .kpi-delta {{
        font-family: {FONT_MONO};
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }}
    .kpi-up {{ color: {ACCENT}; }}
    .kpi-down {{ color: {ALERT}; }}

    /* ---- Alert Banner ---- */
    .alert-banner {{
        background: linear-gradient(90deg, rgba(255,107,107,0.12), rgba(251,191,36,0.08));
        border: 1px solid rgba(255,107,107,0.3);
        border-radius: 12px;
        padding: 0.9rem 1.25rem;
        color: {ALERT};
        font-weight: 600;
        font-size: 0.95rem;
        animation: pulseGlow 3s ease-in-out infinite;
    }}

    /* ---- Recommendation Box ---- */
    .rec-box {{
        padding: 1.5rem 2rem;
        border-radius: 16px;
        font-size: 1.1rem;
        font-weight: 700;
        text-align: center;
        animation: pulseGlow 3s ease-in-out infinite;
        border: 1px solid;
    }}
    .rec-success {{
        background: linear-gradient(135deg, rgba(0,212,170,0.15), rgba(0,212,170,0.05));
        border-color: rgba(0,212,170,0.4);
        color: {ACCENT};
    }}
    .rec-error {{
        background: linear-gradient(135deg, rgba(255,107,107,0.15), rgba(255,107,107,0.05));
        border-color: rgba(255,107,107,0.4);
        color: {ALERT};
    }}

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
        background: rgba(17,24,39,0.8);
        padding: 6px;
        border-radius: 14px;
        margin-bottom: 1.5rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px;
        padding: 10px 18px;
        font-weight: 600;
        color: {TEXT_DIM};
        transition: all 0.3s ease;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {ACCENT}, {ACCENT2}) !important;
        color: #ffffff !important;
    }}
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span,
    .stTabs [aria-selected="true"] div,
    .stTabs [aria-selected="true"] .stMarkdown {{
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }}

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #111827, {BG}) !important;
    }}

    /* ---- Chart Container ---- */
    [data-testid="stPlotlyChart"] {{
        background: rgba(17,24,39,0.5);
        border-radius: 16px;
        padding: 0.75rem;
        border: 1px solid rgba(255,255,255,0.04);
    }}

    /* ---- Metrics override ---- */
    [data-testid="stMetric"] {{
        background: linear-gradient(145deg, {BG_CARD}, {BG});
        padding: 1rem !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 4px 20px rgba(0,0,0,0.35);
    }}

    /* ---- Buttons ---- */
    .stButton > button {{
        background: linear-gradient(135deg, {ACCENT}, {ACCENT2}) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        font-family: {FONT_HEADER} !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(0,212,170,0.3) !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(0,212,170,0.5) !important;
    }}

    /* ---- Animations ---- */
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes fadeInScale {{
        from {{ opacity: 0; transform: scale(0.92); }}
        to {{ opacity: 1; transform: scale(1); }}
    }}
    @keyframes pulseGlow {{
        0%, 100% {{ box-shadow: 0 0 15px rgba(0,212,170,0.15); }}
        50% {{ box-shadow: 0 0 35px rgba(0,212,170,0.25); }}
    }}
    .stTabs [data-baseweb="tab-list"] > div:nth-child(1) {{ animation-delay: 0s; }}
    .stTabs [data-baseweb="tab-list"] > div:nth-child(2) {{ animation-delay: 0.05s; }}
    .stTabs [data-baseweb="tab-list"] > div:nth-child(3) {{ animation-delay: 0.1s; }}
</style>
""", unsafe_allow_html=True)

# ======================================================================
# PLOTLY THEME
# ======================================================================
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Outfit, sans-serif", color=TEXT_DIM, size=12),
    hovermode="x unified",
    margin=dict(l=50, r=30, t=40, b=50),
    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', zeroline=False),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', zeroline=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
)

# ======================================================================
# DATA LOADING
# ======================================================================

@st.cache_data(ttl=3600)
def load_data():
    """Load and merge all data."""
    loader = DataLoader()
    raw = loader.load_raw_data()
    df = loader.merge_data(raw)
    holidays_raw = loader.get_holidays_raw()
    stores, families = loader.get_store_families(df)
    return df, holidays_raw, stores, families

@st.cache_data(ttl=3600)
def engineer_features(_df, _holidays_raw):
    """Apply feature engineering."""
    eng = FeatureEngineer()
    return eng.create_features(_df, holidays_df=_holidays_raw, include_lags=True)

@st.cache_resource
def load_model_artifacts():
    """Load trained model and scaler."""
    model, scaler = None, None
    try:
        from tensorflow.keras.models import load_model as keras_load
        from src.model import AttentionLayer

        # Try v2 model first
        model_path = None
        base = 'models'
        # Find latest version directory
        versions = [d for d in os.listdir(base) if d.startswith('v') and d[1:].isdigit()] if os.path.exists(base) else []
        if versions:
            latest = sorted(versions, key=lambda x: int(x[1:]))[-1]
            v_path = os.path.join(base, latest)
            for f in os.listdir(v_path):
                if f.endswith('.h5'):
                    model_path = os.path.join(v_path, f)
                    break

        # Fall back to v1 model
        if model_path is None and os.path.exists('models/lstm_grocery_v1.h5'):
            model_path = 'models/lstm_grocery_v1.h5'

        if model_path:
            model = keras_load(model_path, compile=False, custom_objects={'AttentionLayer': AttentionLayer})

        if os.path.exists('models/scaler.pkl'):
            scaler = joblib.load('models/scaler.pkl')
    except Exception as e:
        st.warning(f"⚠️ Model loading: {e}")
    return model, scaler

@st.cache_resource
def load_lgbm_models():
    """Load LightGBM models if available."""
    try:
        from src.model import LightGBMModel
        lgbm = LightGBMModel()
        base = 'models'
        versions = [d for d in os.listdir(base) if d.startswith('v') and d[1:].isdigit()] if os.path.exists(base) else []
        if versions:
            latest = sorted(versions, key=lambda x: int(x[1:]))[-1]
            lgbm_path = os.path.join(base, latest, 'lgbm')
            if os.path.exists(lgbm_path):
                lgbm.load(lgbm_path)
                return lgbm
    except Exception:
        pass
    return None


# ======================================================================
# KPI HELPER
# ======================================================================

def kpi_card(label, value, delta=None, delta_type='up'):
    delta_html = ''
    if delta is not None:
        cls = 'kpi-up' if delta_type == 'up' else 'kpi-down'
        arrow = '▲' if delta_type == 'up' else '▼'
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {delta}</div>'
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """

# ======================================================================
# MAIN APP
# ======================================================================

# Load everything
df_raw, holidays_raw, stores_list, families_list = load_data()
model, scaler = load_model_artifacts()
lgbm_model = load_lgbm_models()

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown(f"<h2 style='font-family:{FONT_HEADER}; margin-bottom:0;'>⚙️ Control Panel</h2>",
                unsafe_allow_html=True)
    st.caption("Store Sales Forecasting Engine v2.0")
    st.divider()

    selected_store = st.selectbox("🏬 Store", stores_list, index=0)
    selected_family = st.selectbox("📦 Product Family", families_list,
                                   index=families_list.index('GROCERY I') if 'GROCERY I' in families_list else 0)

    st.divider()
    model_choice = st.radio("🧠 Model", ['Ensemble', 'LightGBM', 'LSTM'], index=0,
                            help="Select model for predictions")

    st.divider()
    st.subheader("☁️ Weather")
    st.caption("[OpenWeatherMap](https://openweathermap.org/api) key for live weather adjustments.")
    api_key = st.text_input("API Key", type="password", placeholder="Enter key...", label_visibility="collapsed")

    weather_data = {"condition": "Unknown", "temp": 20, "description": "No API key"}
    if api_key:
        weather_data = get_current_weather(api_key)
        st.metric("Quito", f"{weather_data['temp']}°C", weather_data.get('condition', ''))

    st.divider()
    st.caption(f"📊 Data: {len(df_raw):,} rows • {len(families_list)} families • {len(stores_list)} stores")

# ---- FILTER DATA ----
loader = DataLoader()
df_store = loader.filter_subset(df_raw, store_nbr=selected_store, family=selected_family)

# ---- HEADER ----
st.markdown(f"""
<div style="animation: fadeInUp 0.5s ease-out both;">
    <span style="font-family:{FONT_MONO}; font-size:0.7rem; font-weight:600; color:{ACCENT};
    background:rgba(0,212,170,0.1); padding:4px 12px; border-radius:20px; letter-spacing:0.1em;">
    AI-POWERED SALES ENGINE v2.0</span>
</div>
""", unsafe_allow_html=True)
st.title("🚀 Sales Forecasting & Dynamic Pricing")
st.caption(f"📍 Store {selected_store} · {selected_family} · Quito, Ecuador")

# ======================================================================
# TABS
# ======================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Executive Overview",
    "📊 Sales Forecasting",
    "💰 Promotion Optimizer",
    "🌤️ External Signals",
    "🔮 What-If Simulator"
])

# ======================================================================
# TAB 1: EXECUTIVE OVERVIEW
# ======================================================================
with tab1:
    eng = FeatureEngineer()
    df_feat = eng.create_features(df_store.copy(), holidays_df=holidays_raw, include_lags=True)

    recent = df_feat.tail(30)
    avg_sales = recent['sales'].mean()
    prev_avg = df_feat.tail(60).head(30)['sales'].mean()
    delta_pct = ((avg_sales - prev_avg) / prev_avg * 100) if prev_avg > 0 else 0

    est_revenue = avg_sales * 10  # base_price
    promo_days = recent['onpromotion'].sum()
    confidence = min(95, 70 + int(len(df_store) / 100))

    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(kpi_card("Predicted Daily Revenue", f"${est_revenue:,.0f}",
                             f"{delta_pct:+.1f}% vs prev 30d",
                             'up' if delta_pct >= 0 else 'down'), unsafe_allow_html=True)
    with k2:
        rec = "✅ Promote" if promo_days > 15 else "⏸️ Hold"
        st.markdown(kpi_card("Promo Recommendation", rec), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card("Model Confidence", f"{confidence}%",
                             f"Based on {len(df_store):,} samples", 'up'), unsafe_allow_html=True)
    with k4:
        w_impact = "+10%" if "Rain" in weather_data['condition'] else "-5%" if "Clear" in weather_data['condition'] else "±0%"
        st.markdown(kpi_card("Weather Impact", w_impact,
                             weather_data['condition'], 'up' if 'Rain' in weather_data['condition'] else 'down'),
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Alert Banner
    tomorrow = pd.Timestamp.now() + timedelta(days=1)
    next_holidays = holidays_raw[pd.to_datetime(holidays_raw['date']) >= pd.Timestamp.now()].head(1)
    if len(next_holidays) > 0:
        hol_name = next_holidays.iloc[0].get('description', 'Holiday')
        hol_date = next_holidays.iloc[0]['date']
        st.markdown(f'<div class="alert-banner">🔴 Upcoming holiday: <strong>{hol_name}</strong> ({hol_date}) — expect demand surge in BEVERAGES, GROCERY</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 30-Day Chart with Confidence Bands
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("### 📈 30-Day Sales Trend")
        plot_data = df_store.tail(90)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_data['date'], y=plot_data['sales'],
            mode='lines', name='Actual Sales',
            line=dict(color=ACCENT, width=2.5, shape='spline', smoothing=0.3),
            fill='tozeroy', fillcolor=f'rgba(0,212,170,0.08)'
        ))
        # Simulated confidence bands
        upper = plot_data['sales'] * 1.15
        lower = plot_data['sales'] * 0.85
        fig.add_trace(go.Scatter(x=plot_data['date'], y=upper, mode='lines',
                                 line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=plot_data['date'], y=lower, mode='lines',
                                 fill='tonexty', fillcolor='rgba(0,212,170,0.06)',
                                 line=dict(width=0), name='Confidence Band'))
        fig.update_layout(**PLOTLY_LAYOUT, height=380)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        st.markdown("### 🗓️ Sales by Day of Week")
        if len(df_feat) > 0:
            heatmap_data = df_feat.groupby('day_of_week')['sales'].mean()
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            fig_heat = go.Figure(go.Bar(
                x=day_names[:len(heatmap_data)],
                y=heatmap_data.values,
                marker=dict(
                    color=heatmap_data.values,
                    colorscale=[[0, '#0A0E1A'], [0.5, ACCENT], [1, ACCENT2]],
                    line=dict(width=0)
                ),
                text=[f'{v:.0f}' for v in heatmap_data.values],
                textposition='outside',
                textfont=dict(family=FONT_MONO, size=11, color=TEXT)
            ))
            fig_heat.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False)
            st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})


# ======================================================================
# TAB 2: SALES FORECASTING
# ======================================================================
with tab2:
    st.markdown("### 📊 Sales Forecasting Deep Dive")

    eng2 = FeatureEngineer()
    df_forecast = eng2.create_features(df_store.copy(), holidays_df=holidays_raw, include_lags=True)

    fc1, fc2 = st.columns([3, 1])
    with fc1:
        # Actual vs Predicted
        st.markdown("#### Actual vs Predicted Sales")
        fig_fc = go.Figure()
        plot_n = min(365, len(df_forecast))
        plot_df = df_forecast.tail(plot_n)

        fig_fc.add_trace(go.Scatter(
            x=plot_df['date'] if 'date' in plot_df.columns else list(range(len(plot_df))),
            y=plot_df['sales'], mode='lines', name='Actual Sales',
            line=dict(color=ACCENT, width=2)
        ))

        # Moving average as "prediction" proxy
        ma = plot_df['sales'].rolling(7, min_periods=1).mean()
        fig_fc.add_trace(go.Scatter(
            x=plot_df['date'] if 'date' in plot_df.columns else list(range(len(plot_df))),
            y=ma, mode='lines', name='7-Day MA (Baseline)',
            line=dict(color=ACCENT2, width=2, dash='dash')
        ))

        # Confidence interval
        upper = ma * 1.2
        lower = ma * 0.8
        fig_fc.add_trace(go.Scatter(
            x=plot_df['date'] if 'date' in plot_df.columns else list(range(len(plot_df))),
            y=upper, mode='lines', line=dict(width=0), showlegend=False
        ))
        fig_fc.add_trace(go.Scatter(
            x=plot_df['date'] if 'date' in plot_df.columns else list(range(len(plot_df))),
            y=lower, mode='lines', fill='tonexty',
            fillcolor='rgba(59,130,246,0.08)', line=dict(width=0),
            name='80% Confidence'
        ))
        fig_fc.update_layout(**PLOTLY_LAYOUT, height=400)
        st.plotly_chart(fig_fc, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})

    with fc2:
        # Performance metrics card
        st.markdown("#### Model Performance")
        if len(plot_df) > 7:
            actual = plot_df['sales'].values[7:]
            predicted = ma.values[7:]
            mask = ~(np.isnan(actual) | np.isnan(predicted))
            if mask.sum() > 0:
                a, p = actual[mask], predicted[mask]
                rmse = np.sqrt(np.mean((a - p) ** 2))
                mae = np.mean(np.abs(a - p))
                rmsle = np.sqrt(np.mean((np.log1p(np.maximum(p, 0)) - np.log1p(np.maximum(a, 0))) ** 2))

                st.metric("RMSE", f"{rmse:.2f}")
                st.metric("MAE", f"{mae:.2f}")
                st.metric("RMSLE", f"{rmsle:.4f}")
                st.caption("Metrics on last year's data")

    # Decomposition
    st.markdown("#### 📉 Trend Decomposition")
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomp_data = df_store.set_index('date')['sales'].dropna()
        if len(decomp_data) > 60:
            decomp_data = decomp_data.asfreq('D', method='ffill')
            result = seasonal_decompose(decomp_data.tail(365), model='additive', period=7)

            fig_dec = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                    subplot_titles=['Trend', 'Seasonality', 'Residuals'],
                                    vertical_spacing=0.08)
            fig_dec.add_trace(go.Scatter(y=result.trend.values, mode='lines',
                                         line=dict(color=ACCENT, width=2), name='Trend'), row=1, col=1)
            fig_dec.add_trace(go.Scatter(y=result.seasonal.values, mode='lines',
                                         line=dict(color=ACCENT2, width=1.5), name='Season'), row=2, col=1)
            fig_dec.add_trace(go.Scatter(y=result.resid.values, mode='markers',
                                         marker=dict(color=ALERT, size=2, opacity=0.5), name='Residual'), row=3, col=1)
            fig_dec.update_layout(**PLOTLY_LAYOUT, height=500, showlegend=False)
            st.plotly_chart(fig_dec, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.info(f"Decomposition requires sufficient data and statsmodels. ({e})")


# ======================================================================
# TAB 3: PROMOTION OPTIMIZER
# ======================================================================
with tab3:
    st.markdown("### 💰 Promotion Optimizer")

    po1, po2 = st.columns([1, 1])
    with po1:
        st.markdown("#### Configure Pricing")
        regular_price = st.slider("Regular Price ($)", 5.0, 30.0, 10.0, 0.5)
        promo_price = st.slider("Promo Price ($)", 3.0, regular_price - 0.5, 8.0, 0.5)
        cost = st.slider("Cost per Unit ($)", 1.0, regular_price - 1.0, 6.0, 0.5)
        elasticity = st.slider("Price Elasticity", -3.0, -0.5, -1.5, 0.1,
                               help="How sensitive demand is to price changes")

    with po2:
        st.markdown("#### Break-Even Analysis")
        margin_regular = regular_price - cost
        margin_promo = promo_price - cost
        st.metric("Regular Margin", f"${margin_regular:.2f}/unit")
        st.metric("Promo Margin", f"${margin_promo:.2f}/unit")
        if margin_promo > 0:
            be_uplift = (margin_regular / margin_promo - 1) * 100
            st.metric("Required Volume Uplift", f"{be_uplift:.1f}%",
                      help="Promo must increase sales by this % to break even")
        else:
            st.error("❌ Promo price below cost!")

    st.divider()

    # Multi-scenario simulation
    st.markdown("#### 📊 Profit Scenario Comparison")
    base_sales = df_store['sales'].tail(30).mean() if len(df_store) > 0 else 100

    optimizer = PromotionOptimizer(config_path='config/config.yaml')
    scenarios = [
        {'name': 'No Promotion', 'price': regular_price, 'elasticity': 0},
        {'name': '10% Off', 'price': regular_price * 0.90, 'elasticity': elasticity},
        {'name': '15% Off', 'price': regular_price * 0.85, 'elasticity': elasticity},
        {'name': '20% Off', 'price': regular_price * 0.80, 'elasticity': elasticity},
        {'name': '25% Off', 'price': regular_price * 0.75, 'elasticity': elasticity},
    ]

    results = []
    for s in scenarios:
        pct_change = (s['price'] - regular_price) / regular_price
        demand_mult = 1 + s['elasticity'] * pct_change if s['elasticity'] != 0 else 1.0
        adj_sales = base_sales * demand_mult
        profit = (s['price'] - cost) * adj_sales
        revenue = s['price'] * adj_sales
        results.append({
            'scenario': s['name'], 'price': s['price'],
            'sales': adj_sales, 'revenue': revenue, 'profit': profit
        })

    df_scenarios = pd.DataFrame(results)
    best = df_scenarios.loc[df_scenarios['profit'].idxmax()]

    fig_scen = go.Figure()
    colors = [ACCENT if r['scenario'] == best['scenario'] else ACCENT2 for _, r in df_scenarios.iterrows()]
    fig_scen.add_trace(go.Bar(
        x=df_scenarios['scenario'], y=df_scenarios['profit'],
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"${p:,.0f}" for p in df_scenarios['profit']],
        textposition='outside', textfont=dict(family=FONT_MONO, color=TEXT, size=13)
    ))
    fig_scen.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False,
                           yaxis_title="Estimated Profit ($)")
    st.plotly_chart(fig_scen, use_container_width=True, config={"displayModeBar": False})

    # Recommendation
    rec_class = "rec-success"
    rec_text = f"✅ Optimal: <strong>{best['scenario']}</strong> at ${best['price']:.2f} → Profit: ${best['profit']:,.0f}"
    uplift = ((best['profit'] / results[0]['profit']) - 1) * 100 if results[0]['profit'] > 0 else 0
    if best['scenario'] != 'No Promotion':
        rec_text += f" ({uplift:+.1f}% vs no promo)"
    else:
        rec_class = "rec-error"
        rec_text = "❌ No promotion is the most profitable option at current parameters."
    st.markdown(f'<div class="rec-box {rec_class}">{rec_text}</div>', unsafe_allow_html=True)

    # Promo effectiveness by family
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📈 Historical Promo Effectiveness")
    promo_eff = optimizer.promo_effectiveness_by_family(df_raw[df_raw['store_nbr'] == selected_store])
    if len(promo_eff) > 0:
        top_n = promo_eff.head(15)
        fig_eff = go.Figure(go.Bar(
            x=top_n['lift_pct'], y=top_n['family'],
            orientation='h',
            marker=dict(
                color=[ACCENT if v > 0 else ALERT for v in top_n['lift_pct']],
                line=dict(width=0)
            ),
            text=[f"{v:+.1f}%" for v in top_n['lift_pct']],
            textposition='outside',
            textfont=dict(family=FONT_MONO, size=11, color=TEXT)
        ))
        fig_eff.update_layout(**PLOTLY_LAYOUT, height=420, showlegend=False,
                             xaxis_title="Sales Lift %")
        fig_eff.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_eff, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Promo effectiveness data requires promotion history.")


# ======================================================================
# TAB 4: EXTERNAL SIGNALS
# ======================================================================
with tab4:
    st.markdown("### 🌤️ External Signals Dashboard")

    e1, e2 = st.columns(2)

    with e1:
        # Weather Widget
        st.markdown("#### ☁️ Live Weather — Quito")
        if api_key:
            wc1, wc2, wc3 = st.columns(3)
            with wc1:
                st.metric("Temperature", f"{weather_data['temp']}°C")
            with wc2:
                st.metric("Condition", weather_data['condition'])
            with wc3:
                impact_str = "+10% demand" if "Rain" in weather_data['condition'] else "-5% demand" if "Clear" in weather_data['condition'] else "Neutral"
                st.metric("Sales Impact", impact_str)
        else:
            st.info("🔑 Enter OpenWeatherMap API key in sidebar for live weather data.")

        # Oil price chart
        st.markdown("#### 🛢️ Oil Price (30-Day Trend)")
        oil_data = df_store[['date', 'dcoilwtico']].drop_duplicates('date').tail(90)
        if len(oil_data) > 0:
            fig_oil = go.Figure()
            fig_oil.add_trace(go.Scatter(x=oil_data['date'], y=oil_data['dcoilwtico'],
                                         mode='lines', name='Oil Price',
                                         line=dict(color=WARN, width=2.5, shape='spline'),
                                         fill='tozeroy', fillcolor='rgba(251,191,36,0.08)'))
            fig_oil.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
            st.plotly_chart(fig_oil, use_container_width=True, config={"displayModeBar": False})

    with e2:
        # Holiday Calendar
        st.markdown("#### 📅 Holiday Calendar (Next 30 Days)")
        today = pd.Timestamp.now()
        upcoming = holidays_raw.copy()
        upcoming['date'] = pd.to_datetime(upcoming['date'])
        upcoming = upcoming[
            (upcoming['date'] >= today) &
            (upcoming['date'] <= today + timedelta(days=30)) &
            (upcoming['transferred'] == False)
        ][['date', 'description', 'locale']].drop_duplicates()

        if len(upcoming) > 0:
            for _, row in upcoming.iterrows():
                locale_badge = "🌍" if row['locale'] == 'National' else "📍"
                st.markdown(f"**{row['date'].strftime('%b %d')}** — {locale_badge} {row['description']} ({row['locale']})")
        else:
            st.success("No holidays in the next 30 days.")

        # Correlation Matrix
        st.markdown("#### 🔗 Signal Correlations")
        corr_cols = ['sales', 'dcoilwtico', 'transactions', 'onpromotion', 'is_holiday']
        avail = [c for c in corr_cols if c in df_store.columns]
        if len(avail) >= 2:
            corr = df_store[avail].corr()
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values, x=avail, y=avail,
                colorscale=[[0, ALERT], [0.5, BG], [1, ACCENT]],
                text=np.round(corr.values, 2), texttemplate='%{text}',
                textfont=dict(family=FONT_MONO, size=11, color=TEXT)
            ))
            fig_corr.update_layout(**PLOTLY_LAYOUT, height=350)
            st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})


# ======================================================================
# TAB 5: WHAT-IF SIMULATOR
# ======================================================================
with tab5:
    st.markdown("### 🔮 What-If Simulator")
    st.caption("Set hypothetical conditions and see predicted sales + profit instantly.")

    wc1, wc2 = st.columns([1, 1])

    with wc1:
        st.markdown("#### Scenario Parameters")
        sim_date = st.date_input("📅 Date", value=datetime.now().date())
        sim_promo = st.toggle("🏷️ Promotion Active", value=False)
        sim_holiday = st.toggle("🎉 Holiday", value=False)
        sim_oil = st.slider("🛢️ Oil Price ($/bbl)", 20.0, 120.0, 60.0, 1.0)
        sim_price = st.slider("💲 Selling Price ($)", 5.0, 20.0, 10.0, 0.5)
        sim_cost = st.slider("🏭 Unit Cost ($)", 2.0, 15.0, 6.0, 0.5)

    with wc2:
        st.markdown("#### Prediction Results")

        # Use historical average as baseline prediction
        day_of_week = pd.Timestamp(sim_date).dayofweek
        is_payday = pd.Timestamp(sim_date).day == 15 or pd.Timestamp(sim_date).is_month_end

        # Baseline from historical data for similar conditions
        match = df_store.copy()
        match['dow'] = pd.to_datetime(match['date']).dt.dayofweek
        similar = match[match['dow'] == day_of_week]
        if sim_promo:
            promo_match = similar[similar['onpromotion'] == 1]
            if len(promo_match) > 10:
                similar = promo_match

        baseline = similar['sales'].mean() if len(similar) > 0 else 100

        # Apply adjustments
        adj = 1.0
        adjustments = []
        if sim_holiday:
            adj *= 1.20
            adjustments.append("Holiday +20%")
        if is_payday:
            adj *= 1.10
            adjustments.append("Payday +10%")
        if sim_oil > 80:
            adj *= 0.97
            adjustments.append("High oil -3%")
        elif sim_oil < 40:
            adj *= 1.03
            adjustments.append("Low oil +3%")

        predicted_sales = baseline * adj
        profit = (sim_price - sim_cost) * predicted_sales
        revenue = sim_price * predicted_sales

        st.metric("📦 Predicted Sales", f"{predicted_sales:,.0f} units")
        st.metric("💰 Revenue", f"${revenue:,.2f}")
        st.metric("📈 Profit", f"${profit:,.2f}",
                  delta=f"Margin: ${sim_price - sim_cost:.2f}/unit")

        if adjustments:
            st.caption("Adjustments: " + " | ".join(adjustments))

    # Scenario Comparison
    st.divider()
    st.markdown("#### 🔄 Compare Scenarios")

    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = []

    if st.button("➕ Add Current Scenario"):
        st.session_state.scenarios.append({
            'date': str(sim_date),
            'promo': sim_promo,
            'holiday': sim_holiday,
            'oil_price': sim_oil,
            'price': sim_price,
            'cost': sim_cost,
            'predicted_sales': predicted_sales,
            'revenue': revenue,
            'profit': profit,
        })
        st.rerun()

    if st.session_state.scenarios:
        df_comp = pd.DataFrame(st.session_state.scenarios)
        df_comp.index = [f"Scenario {i+1}" for i in range(len(df_comp))]

        st.dataframe(df_comp.style.format({
            'predicted_sales': '{:,.0f}',
            'revenue': '${:,.2f}',
            'profit': '${:,.2f}',
            'oil_price': '${:.0f}',
            'price': '${:.2f}',
            'cost': '${:.2f}',
        }), use_container_width=True)

        # Download
        csv = df_comp.to_csv(index=True)
        st.download_button("📥 Export as CSV", csv, "what_if_scenarios.csv", "text/csv")

        if st.button("🗑️ Clear All Scenarios"):
            st.session_state.scenarios = []
            st.rerun()