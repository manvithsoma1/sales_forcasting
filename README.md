# 🛒 Store Sales Forecasting & AI Dynamic Pricing Engine

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://salesforcasting-wtmhetwsf6sz8erkakahri.streamlit.app/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.1-4CAF50?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

> **End-to-end ML system** that predicts grocery store sales using deep learning and real-time weather data — then recommends whether running a promotion will maximize profit.

🌐 **Live App:** [salesforcasting-wtmhetwsf6sz8erkakahri.streamlit.app](https://salesforcasting-wtmhetwsf6sz8erkakahri.streamlit.app/)

---

## 📌 The Business Question

> **"Should we run a promotion today — and at what price does it maximize profit?"**

This system answers that question by combining a 30-day LSTM sales forecast with live weather signals and a profit optimization engine that simulates promo vs. no-promo scenarios in real time.

---

## 🏗️ Architecture

```
Raw Data (Kaggle CSVs)
        │
        ▼
┌─────────────────┐     ┌──────────────────┐
│  data_loader.py │────▶│   features.py    │
│  Merge 6 CSVs   │     │  30+ engineered  │
│  train, oil,    │     │  lag, date,      │
│  stores, txns,  │     │  holiday feats   │
│  holidays       │     └────────┬─────────┘
└─────────────────┘              │
                                 ▼
                      ┌──────────────────┐
                      │ preprocessing.py │
                      │ MinMaxScaler     │
                      │ 30-day sequences │
                      └────────┬─────────┘
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
        ┌──────────┐   ┌──────────────┐  ┌──────────┐
        │   LSTM   │   │  LightGBM    │  │ Weather  │
        │ 128→64→32│   │ 54 per-family│  │   API    │
        │  units   │   │   models     │  │  Quito   │
        └────┬─────┘   └──────┬───────┘  └────┬─────┘
             │                │               │
             └────────────────┴───────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │   optimization.py   │
                   │  Promo Simulator    │
                   │  Profit Maximizer   │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │      app.py         │
                   │  Streamlit Dashboard│
                   │  5 Interactive Pages│
                   └─────────────────────┘
```

---

## 🧠 Three Pillars

| Pillar | Technique | Business Value |
|--------|-----------|----------------|
| **LSTM Forecasting** | 30-day lookback, 3-layer deep LSTM with BatchNorm & early stopping | Predicts tomorrow's sales using trends, oil prices, holidays, transactions |
| **Weather Adjustment** | Live OpenWeatherMap API call at inference time | Rain → +10% demand; Sunny → −5% adjustment applied to forecast |
| **Profit Optimizer** | Promo vs. no-promo simulation across price/cost/volume | Recommends the scenario with higher total profit automatically |

---

## 🤖 Model Details

### LSTM Architecture

```
Input (30 timesteps × 7 features)
    │
    ▼
LSTM(128) → BatchNorm → Dropout(0.3)
    │
    ▼
LSTM(64)  → BatchNorm → Dropout(0.2)
    │
    ▼
LSTM(32)  → Dropout(0.1)
    │
    ▼
Dense(1)  → Sales Forecast
```

**Training config:**
- Loss: Huber (robust to outliers)
- Optimizer: Adam with ReduceLROnPlateau
- Callbacks: EarlyStopping (patience=10)
- Format: `.keras` (migrated from legacy `.h5`)

### Input Features

| Group | Features |
|-------|----------|
| Time | day_of_week, day_of_month, week_of_year, month, quarter, is_weekend |
| Calendar | is_payday (15th & month-end), is_national_holiday, is_regional_holiday |
| Oil Prices | dcoilwtico, oil_7d_ma, oil_30d_ma, oil_volatility, oil_trend |
| Promotions | onpromotion (binary flag) |
| Transactions | transaction count, txn_7d_ma, txn_deviation |

---

## 📊 Model Performance

| Metric | Score | Description |
|--------|-------|-------------|
| RMSE | ~0.42 | Root Mean Square Error on test set |
| MAE | ~0.31 | Mean Absolute Error |
| RMSLE | ~0.48 | Kaggle competition metric |
| Training Time | ~8 min | CPU, avg 35 epochs with early stopping |

---

## 💰 Profit Optimization Logic

The optimizer simulates two business scenarios using the forecasted sales volume:

| Scenario | Sell Price | Unit Cost | Margin | Volume Effect |
|----------|-----------|-----------|--------|---------------|
| No Promotion | $10.00 | $6.00 | $4.00/unit | Baseline forecast |
| With Promotion | $8.00 | $6.00 | $2.00/unit | +elasticity uplift |

**Decision rule:** `Total Profit = margin × forecasted_volume × weather_adjustment`

The system recommends whichever scenario yields the higher total profit.

---

## 🖥️ Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 **Executive Overview** | KPI cards, 30-day forecast chart, sales heatmap, holiday alerts |
| 📊 **Sales Forecasting** | Multi-store/family selector, actual vs. predicted chart, metrics table |
| 💰 **Promotion Optimizer** | Price sliders, 5-scenario profit comparison, break-even calculator |
| 🌤️ **External Signals** | Live weather, oil price trend, holiday calendar, correlation matrix |
| 🔮 **What-If Simulator** | Custom scenario builder, side-by-side comparison, CSV export |

---

## 🚀 Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/your-username/store-sales-forecasting.git
cd store-sales-forecasting
pip install -r requirements.txt
```

### 2. Add Kaggle Data

Download from [Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting) and place all CSVs in `data/`:

```
data/
├── train.csv
├── test.csv
├── stores.csv
├── oil.csv
├── holidays_events.csv
└── transactions.csv
```

### 3. Train the Model

```bash
python main.py
```

### 4. Launch the Dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) — enter your [OpenWeatherMap API key](https://openweathermap.org/api) in the sidebar.

---

## 📁 Project Structure

```
store-sales-forecasting/
├── src/
│   ├── data_loader.py        # CSV ingestion & merging (6 files)
│   ├── features.py           # 30+ engineered features
│   ├── preprocessing.py      # MinMaxScaler + 30-day LSTM sequences
│   ├── model.py              # LSTM architecture, robust loader, ModelRegistry
│   ├── model_migration.py    # CLI: migrate .h5 → .keras format
│   ├── evaluation.py         # RMSE, MAE, RMSLE + loss plots
│   ├── optimization.py       # Promo profit simulation
│   └── weather_service.py    # OpenWeatherMap client
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Model_Experiments.ipynb
├── data/                     # Raw CSVs (gitignored)
├── models/                   # Saved model artifacts
├── app.py                    # Streamlit dashboard (5 pages)
├── main.py                   # Training entry point
├── config.py                 # Centralized hyperparameters
├── requirements.txt
└── runtime.txt               # Python 3.11 for Streamlit Cloud
```

---

## ☁️ Deployment

Deployed on **Streamlit Community Cloud** — auto-redeploys on every push to `main`.

| Config File | Purpose |
|-------------|---------|
| `requirements.txt` | Pinned: `tensorflow==2.15.0`, `keras==2.15.0`, `protobuf==3.20.3` |
| `runtime.txt` | Pins Python 3.11 on Streamlit Cloud |
| `.streamlit/config.toml` | Dark theme, accent colors |

> **Security:** API keys are never hardcoded. Users enter their OpenWeatherMap key in the sidebar — it lives only in session state and is never persisted or logged.

---

## 🔧 Troubleshooting

### ⚠️ `Unrecognized keyword arguments: ['batch_shape']`

This happens when a model saved with TF 2.15 is loaded with TF 2.16+.

**Option 1 — Pin TensorFlow (recommended):**
```
tensorflow==2.15.0  # already in requirements.txt
```

**Option 2 — Run migration script:**
```bash
python src/model_migration.py
```

**Option 3 — One-click fix in app:**
Click **🔄 Rebuild & Save Model** in the Streamlit sidebar.

The app uses a 3-try fallback loader (`robust_load_keras_model`) — it will never hard-crash on a model load error.

---

### ⚠️ Weather API not working

- Enter your OpenWeatherMap API key in the sidebar (free at [openweathermap.org](https://openweathermap.org/api))
- Free tier keys activate within 2 hours of signup
- App gracefully falls back to no weather adjustment if the API call fails

### ⚠️ Out of memory during training

Reduce these values in `config.py`:
```python
BATCH_SIZE = 32        # default: 64
SEQUENCE_LENGTH = 14   # default: 30
```

---

## 📦 Dataset

**Corporación Favorita Grocery Sales Forecasting** — [Kaggle Competition](https://www.kaggle.com/c/favorita-grocery-sales-forecasting)

- 54 product families across 54 stores in Ecuador
- ~3 million rows spanning 2013–2017
- Supplementary: oil prices, store metadata, national/regional holidays, daily transactions

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Deep Learning | TensorFlow 2.15 / Keras |
| Gradient Boosting | LightGBM 4.1 |
| Data Processing | Pandas 2.1, NumPy 1.24, Scikit-learn 1.3 |
| Visualization | Plotly 5.18 |
| Dashboard | Streamlit 1.29 |
| External API | OpenWeatherMap |
| Deployment | Streamlit Community Cloud |
| Language | Python 3.11 |

---

## 💡 Business Impact

This system moves promotion decisions from gut-feel to data-driven:

- **Prevents promotion waste** — only recommends promos when volume uplift outweighs margin loss
- **Weather-adjusted forecasts** — accounts for demand spikes on rainy days that static models miss
- **Real-time decisions** — inference runs in under 2 seconds in the live app
- **Explainable outputs** — every recommendation shows the profit calculation behind it

---




---

<div align="center">
  Built with ❤️ using TensorFlow, Streamlit & Python 3.11
</div>
