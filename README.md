# netflix-stock-forecasting[README.md](https://github.com/user-attachments/files/26104673/README.md)
# Netflix Stock Intelligence Dashboardgh

An interactive stock analysis and forecasting dashboard built with Streamlit and Facebook Prophet. Covers technical analysis, momentum indicators, and time-series forecasting on NFLX historical price data (2002–2026).

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red) ![Prophet](https://img.shields.io/badge/Prophet-1.1-orange)

---

## Features

- **Price Chart** — Close price with optional 30/90-day moving averages and Bollinger Bands (20d, 2σ)
- **Volume Analysis** — Color-coded volume bars tied to daily return direction
- **RSI** — 14-day Relative Strength Index with overbought/oversold zones
- **MACD** — EMA(12/26) crossover with signal line and histogram
- **Prophet Forecasting** — Multiplicative seasonality model with custom quarterly cycle; evaluated on a held-out 90-day test set
- **Model Metrics** — RMSE, MAE, and MAPE reported against actual test prices
- **Executive Summary** — Auto-generated narrative interpreting current indicator signals
- **Sidebar Controls** — Toggle indicators, adjust forecast horizon (30–365 days), and filter by date range

---

## Tech Stack

| Layer | Library |
|---|---|
| UI / App | Streamlit |
| Data | Pandas, NumPy |
| Visualization | Matplotlib |
| Forecasting | Prophet (Meta) |
| Evaluation | scikit-learn |

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/netflix-stock-intelligence.git
cd netflix-stock-intelligence
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** Prophet requires `pystan`. If installation fails, try:
> ```bash
> pip install pystan==2.19.1.1
> pip install prophet
> ```

### 3. Run the app
```bash
streamlit run app.py
```

Make sure `NFLX.csv` is in the same directory as `app.py`.

---

## Data

`NFLX.csv` contains daily OHLCV data for Netflix (NASDAQ: NFLX) from May 2002 to January 2026, sourced from Yahoo Finance.

Columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`

---

## Model Details

The forecasting model uses **Facebook Prophet** with the following configuration:

- **Mode:** Multiplicative seasonality (better for stocks with growth trends)
- **Changepoint prior scale:** 0.05 (moderate flexibility)
- **Custom seasonality:** Quarterly (91.25-day period, Fourier order 5)
- **Train/test split:** All data minus last 90 days for training; final 90 days held out for evaluation
- **Evaluation:** `merge_asof` used to align Prophet's calendar (includes weekends) with market trading days before computing metrics

---

## Project Structure

```
├── app.py              # Main Streamlit application
├── NFLX.csv            # Historical price data
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Disclaimer

This dashboard is for educational and analytical purposes only. Nothing here constitutes financial advice.
