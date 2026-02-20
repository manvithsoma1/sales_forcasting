# Store Sales Forecasting & AI Dynamic Pricing Engine

AI-powered store sales forecasting and promotion optimization. Uses **LSTM** for time-series predictions and **real-time weather** to recommend when to run promotions.

## Objective

Maximize profit by answering: **"Should we run a promotion today?"**

- **LSTM model** — Predicts sales from the last 30 days (oil prices, holidays, transactions, promotions)
- **Weather rules** — Rain (+10% demand), sunny (−5% demand) adjust the baseline
- **Profit comparison** — No promo vs promo scenario with margin math

## Project Structure

```
store-sales-forecasting/
├── app.py              # Streamlit web UI
├── main.py             # Train model & run optimization
├── config/config.yaml  # Data paths, model hyperparameters
├── src/
│   ├── data_loader.py  # Load & merge train, oil, stores, holidays, transactions
│   ├── features.py     # Date features (weekend, payday), model columns
│   ├── preprocessing.py# Scale data, create LSTM sequences
│   ├── model.py        # LSTM architecture (early stopping, BatchNorm)
│   ├── evaluation.py   # RMSE, MAE, loss plots
│   ├── optimization.py # Promo vs no-promo profit simulation
│   └── weather_service.py  # OpenWeatherMap API (Quito)
├── data/raw/           # train.zip or train.csv, oil.csv, stores.csv, etc.
└── models/             # lstm_grocery_v1.h5, scaler.pkl
```

## Setup

1. **Clone & install**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add data** (Corporación Favorita dataset from Kaggle)
   - Place `train.zip` (or `train.csv`), `oil.csv`, `stores.csv`, `holidays_events.csv`, `transactions.csv` in `data/raw/`

3. **Train model**
   ```bash
   python main.py
   ```
   Trains an LSTM on Store 1, Grocery I, saves model and scaler to `models/`.

4. **Run app**
   ```bash
   streamlit run app.py
   ```

5. **Optional:** Add [OpenWeatherMap](https://openweathermap.org/api) API key in the sidebar for weather-adjusted recommendations.

## How It Works

- **Input:** Last 30 days of sales, oil, holidays, transactions, promotions
- **Output:** Predicted sales for “tomorrow” under two scenarios (no promo vs promo)
- **Weather:** Real-time Quito weather adjusts demand before profit comparison
- **Business:** Base price $10, promo $8 (20% off), cost $6 → profit per unit

## Deploying to Streamlit Cloud

1. **Choose Python 3.11** in Advanced settings when deploying (required for TensorFlow compatibility).
2. `requirements.txt` pins `protobuf==3.20.3` and `tensorflow==2.15.0` to avoid the "Descriptors cannot be created" error.
3. **If the error persists**, add this in App → Settings → Advanced → Environment variables:
   - `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` = `python`

## Improvements Made

- **Bug fixes:** Correct `onpromotion` column index (1), app indentation, transactions merge
- **Model:** Deeper LSTM (128→64→32), BatchNorm, early stopping, ReduceLROnPlateau
- **UI:** Clean layout, tooltips, clearer tabs, modern chart styling
- **Security:** API key input only (no hardcoded keys)
