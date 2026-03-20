import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Stock Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS — Professional Dark Theme
# ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #0d0d0d;
        color: #e8e8e8;
    }
    .metric-card {
        background: #161616;
        border: 1px solid #2a2a2a;
        border-left: 3px solid #e50914;
        border-radius: 4px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .metric-label {
        font-size: 11px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #888;
        font-family: 'IBM Plex Mono', monospace;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 600;
        color: #f0f0f0;
        font-family: 'IBM Plex Mono', monospace;
    }
    .metric-delta {
        font-size: 12px;
        font-family: 'IBM Plex Mono', monospace;
        margin-top: 4px;
    }
    .section-header {
        font-size: 13px;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #e50914;
        font-family: 'IBM Plex Mono', monospace;
        border-bottom: 1px solid #2a2a2a;
        padding-bottom: 8px;
        margin: 28px 0 16px 0;
    }
    .insight-box {
        background: #111;
        border: 1px solid #222;
        border-radius: 4px;
        padding: 16px 20px;
        font-size: 14px;
        line-height: 1.7;
        color: #ccc;
    }
    .caption {
        font-size: 12px;
        color: #666;
        font-style: italic;
        margin-top: 4px;
    }
    h1 { font-family: 'IBM Plex Sans', sans-serif; font-weight: 300; letter-spacing: -0.5px; }
    .stSidebar { background-color: #111 !important; }
    .stMetric { background: #161616; border-radius: 4px; padding: 12px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d0d0d",
    "axes.facecolor": "#0d0d0d",
    "axes.edgecolor": "#2a2a2a",
    "axes.labelcolor": "#888",
    "xtick.color": "#555",
    "ytick.color": "#555",
    "grid.color": "#1f1f1f",
    "text.color": "#ccc",
    "legend.facecolor": "#161616",
    "legend.edgecolor": "#2a2a2a",
    "font.family": "monospace",
})

@st.cache_data
def load_data(path="NFLX.csv"):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

@st.cache_data
def compute_indicators(df):
    df = df.copy()
    df['MA30']     = df['Close'].rolling(30).mean()
    df['MA90']     = df['Close'].rolling(90).mean()
    df['Returns']  = df['Close'].pct_change()
    df['CumReturn']= (1 + df['Returns']).cumprod() - 1
    df['Volatility_30d'] = df['Returns'].rolling(30).std() * np.sqrt(252)

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss  = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_MID']   = df['Close'].rolling(20).mean()
    df['BB_STD']   = df['Close'].rolling(20).std()
    df['BB_UPPER'] = df['BB_MID'] + 2 * df['BB_STD']
    df['BB_LOWER'] = df['BB_MID'] - 2 * df['BB_STD']
    df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MID']

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']

    return df

@st.cache_data
def run_prophet(df, forecast_days):
    prophet_df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    train = prophet_df.iloc[:-90]
    test  = prophet_df.iloc[-90:]

    model = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_mode='multiplicative',
        yearly_seasonality=True,
        weekly_seasonality=False
    )
    model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
    model.fit(train)

    future   = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    # Merge on nearest date to handle weekends/holidays missing from Prophet's calendar
    forecast_indexed = forecast[['ds', 'yhat']].set_index('ds').sort_index()
    test_merged = pd.merge_asof(
        test.sort_values('ds'),
        forecast_indexed.reset_index(),
        on='ds',
        direction='nearest'
    )
    rmse = sqrt(mean_squared_error(test_merged['y'].values, test_merged['yhat'].values))
    mae  = mean_absolute_error(test_merged['y'].values, test_merged['yhat'].values)
    mape = np.mean(np.abs((test_merged['y'].values - test_merged['yhat'].values) / test_merged['y'].values)) * 100

    return model, forecast, rmse, mae, mape

# ─────────────────────────────────────────
# LOAD & SIDEBAR
# ─────────────────────────────────────────
df_raw = load_data()

st.sidebar.markdown("### ⚙️ Controls")
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 30, 365, 90, step=15)
show_ma       = st.sidebar.checkbox("Moving Averages (30/90d)", True)
show_bbands   = st.sidebar.checkbox("Bollinger Bands", True)
show_rsi      = st.sidebar.checkbox("RSI (14d)", True)
show_macd     = st.sidebar.checkbox("MACD", True)
show_volume   = st.sidebar.checkbox("Volume Analysis", True)

date_range = st.sidebar.date_input(
    "Date Range",
    [df_raw['Date'].min(), df_raw['Date'].max()]
)

df = df_raw[
    (df_raw['Date'] >= pd.to_datetime(date_range[0])) &
    (df_raw['Date'] <= pd.to_datetime(date_range[1]))
].copy()

df = compute_indicators(df)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("# Netflix Stock Intelligence")
st.markdown('<p class="caption">Technical analysis · Forecasting · Business insights</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────
st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)

latest     = df.iloc[-1]
prev_year  = df[df['Date'] <= df['Date'].max() - pd.DateOffset(years=1)].iloc[-1]
yoy_return = (latest['Close'] / prev_year['Close'] - 1) * 100
total_return = df['CumReturn'].iloc[-1] * 100
avg_vol    = df['Volatility_30d'].dropna().mean() * 100
high_52w   = df[df['Date'] >= df['Date'].max() - pd.DateOffset(days=365)]['High'].max()
low_52w    = df[df['Date'] >= df['Date'].max() - pd.DateOffset(days=365)]['Low'].min()

def card(label, value, delta=None, delta_positive=None):
    color = "#2ecc71" if delta_positive else "#e74c3c" if delta_positive is False else "#888"
    delta_html = f'<div class="metric-delta" style="color:{color}">{delta}</div>' if delta else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>"""

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.markdown(card("Latest Close", f"${latest['Close']:.2f}"), unsafe_allow_html=True)
with c2: st.markdown(card("YoY Return", f"{yoy_return:+.1f}%", delta="vs 1 year ago", delta_positive=yoy_return > 0), unsafe_allow_html=True)
with c3: st.markdown(card("Period Return", f"{total_return:+.1f}%"), unsafe_allow_html=True)
with c4: st.markdown(card("Avg Ann. Volatility", f"{avg_vol:.1f}%"), unsafe_allow_html=True)
with c5: st.markdown(card("52W Range", f"${low_52w:.0f} – ${high_52w:.0f}"), unsafe_allow_html=True)

# ─────────────────────────────────────────
# PRICE CHART
# ─────────────────────────────────────────
st.markdown('<div class="section-header">Price Chart</div>', unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(df['Date'], df['Close'], color='#e50914', linewidth=1.2, label='Close Price', zorder=3)

if show_ma:
    ax.plot(df['Date'], df['MA30'], color='#3498db', linewidth=0.9, linestyle='--', label='MA 30d', alpha=0.8)
    ax.plot(df['Date'], df['MA90'], color='#9b59b6', linewidth=0.9, linestyle='--', label='MA 90d', alpha=0.8)

if show_bbands:
    ax.fill_between(df['Date'], df['BB_LOWER'], df['BB_UPPER'], alpha=0.08, color='#f39c12', label='Bollinger Bands')
    ax.plot(df['Date'], df['BB_UPPER'], color='#f39c12', linewidth=0.5, linestyle=':')
    ax.plot(df['Date'], df['BB_LOWER'], color='#f39c12', linewidth=0.5, linestyle=':')

ax.set_ylabel("Price (USD)")
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)
st.markdown('<p class="caption">Bollinger Bands (20d, 2σ) show price volatility bands. MA crossovers indicate trend shifts.</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# VOLUME (optional)
# ─────────────────────────────────────────
if show_volume and 'Volume' in df.columns:
    st.markdown('<div class="section-header">Volume Analysis</div>', unsafe_allow_html=True)
    fig_v, (ax_v1, ax_v2) = plt.subplots(2, 1, figsize=(13, 5), gridspec_kw={'height_ratios': [2, 1]})

    colors = ['#2ecc71' if r >= 0 else '#e74c3c' for r in df['Returns'].fillna(0)]
    ax_v1.plot(df['Date'], df['Close'], color='#e50914', linewidth=1.0)
    ax_v1.set_ylabel("Price (USD)")
    ax_v1.grid(True, alpha=0.2)

    ax_v2.bar(df['Date'], df['Volume'], color=colors, alpha=0.7, width=1.0)
    ax_v2.set_ylabel("Volume")
    ax_v2.grid(True, alpha=0.2)

    plt.tight_layout()
    st.pyplot(fig_v)
    st.markdown('<p class="caption">Volume bars colored green (up day) or red (down day). High-volume moves signal institutional participation.</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# RSI
# ─────────────────────────────────────────
if show_rsi:
    st.markdown('<div class="section-header">RSI — Momentum Indicator</div>', unsafe_allow_html=True)
    fig_rsi, ax_rsi = plt.subplots(figsize=(13, 3))
    ax_rsi.plot(df['Date'], df['RSI'], color='#f1c40f', linewidth=1.0)
    ax_rsi.fill_between(df['Date'], df['RSI'], 70, where=(df['RSI'] >= 70), color='#e74c3c', alpha=0.25, label='Overbought')
    ax_rsi.fill_between(df['Date'], df['RSI'], 30, where=(df['RSI'] <= 30), color='#2ecc71', alpha=0.25, label='Oversold')
    ax_rsi.axhline(70, color='#e74c3c', linestyle='--', linewidth=0.7)
    ax_rsi.axhline(30, color='#2ecc71', linestyle='--', linewidth=0.7)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.legend(fontsize=9)
    ax_rsi.grid(True, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig_rsi)
    st.markdown('<p class="caption">RSI > 70 = overbought (potential reversal). RSI < 30 = oversold (potential bounce). Use as confirmation, not standalone signal.</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# MACD
# ─────────────────────────────────────────
if show_macd:
    st.markdown('<div class="section-header">MACD — Trend Momentum</div>', unsafe_allow_html=True)
    fig_m, ax_m = plt.subplots(figsize=(13, 3))
    ax_m.plot(df['Date'], df['MACD'],        color='#3498db', linewidth=1.0, label='MACD')
    ax_m.plot(df['Date'], df['MACD_Signal'], color='#e67e22', linewidth=1.0, label='Signal')
    hist_colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in df['MACD_Hist']]
    ax_m.bar(df['Date'], df['MACD_Hist'], color=hist_colors, alpha=0.5, width=1.0, label='Histogram')
    ax_m.axhline(0, color='#555', linewidth=0.5)
    ax_m.legend(fontsize=9)
    ax_m.grid(True, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig_m)
    st.markdown('<p class="caption">MACD crossing above signal line = bullish momentum. Histogram bars above zero confirm upward trend strength.</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# RETURNS DISTRIBUTION
# ─────────────────────────────────────────
st.markdown('<div class="section-header">Returns Distribution</div>', unsafe_allow_html=True)
fig_r, (ax_r1, ax_r2) = plt.subplots(1, 2, figsize=(13, 4))

returns = df['Returns'].dropna() * 100
ax_r1.hist(returns, bins=60, color='#e50914', alpha=0.7, edgecolor='none')
ax_r1.axvline(0, color='white', linewidth=0.8, linestyle='--')
ax_r1.axvline(returns.mean(), color='#f1c40f', linewidth=1.0, linestyle='--', label=f'Mean {returns.mean():.2f}%')
ax_r1.set_xlabel("Daily Return (%)")
ax_r1.set_ylabel("Frequency")
ax_r1.set_title("Daily Returns Distribution")
ax_r1.legend(fontsize=9)
ax_r1.grid(True, alpha=0.2)

ax_r2.plot(df['Date'], df['CumReturn'] * 100, color='#2ecc71', linewidth=1.1)
ax_r2.axhline(0, color='#555', linewidth=0.5)
ax_r2.fill_between(df['Date'], df['CumReturn'] * 100, 0,
                   where=(df['CumReturn'] >= 0), color='#2ecc71', alpha=0.1)
ax_r2.fill_between(df['Date'], df['CumReturn'] * 100, 0,
                   where=(df['CumReturn'] < 0), color='#e74c3c', alpha=0.1)
ax_r2.set_xlabel("Date")
ax_r2.set_ylabel("Cumulative Return (%)")
ax_r2.set_title("Cumulative Return")
ax_r2.grid(True, alpha=0.2)

plt.tight_layout()
st.pyplot(fig_r)
st.markdown(f'<p class="caption">Annualized volatility: {avg_vol:.1f}%. Skew: {returns.skew():.2f} | Kurtosis: {returns.kurt():.2f} — higher kurtosis indicates fat tails and crash risk.</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# PROPHET FORECASTING
# ─────────────────────────────────────────
st.markdown('<div class="section-header">Prophet Forecast Model</div>', unsafe_allow_html=True)

with st.spinner("Training forecast model..."):
    model, forecast, rmse, mae, mape = run_prophet(df, forecast_days)

col_f1, col_f2 = st.columns([3, 1])

with col_f1:
    fig_fc, ax_fc = plt.subplots(figsize=(10, 5))
    hist_dates = forecast[forecast['ds'] <= df['Date'].max()]
    fut_dates  = forecast[forecast['ds'] > df['Date'].max()]

    ax_fc.plot(df['Date'], df['Close'], color='#e50914', linewidth=1.0, label='Actual', zorder=3)
    ax_fc.plot(forecast['ds'], forecast['yhat'], color='#3498db', linewidth=1.0, linestyle='--', label='Forecast', alpha=0.8)
    ax_fc.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                       alpha=0.12, color='#3498db', label='80% Confidence Interval')
    ax_fc.axvline(df['Date'].max(), color='#f1c40f', linewidth=0.8, linestyle=':', label='Forecast Start')
    ax_fc.set_ylabel("Price (USD)")
    ax_fc.legend(fontsize=9)
    ax_fc.grid(True, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig_fc)

with col_f2:
    st.markdown("**Model Evaluation**")
    st.markdown(f"""
    <div class="insight-box">
    <div class="metric-label">RMSE</div>
    <div style="font-size:22px;font-family:monospace;color:#f1c40f;">${rmse:.2f}</div>
    <br>
    <div class="metric-label">MAE</div>
    <div style="font-size:22px;font-family:monospace;color:#f1c40f;">${mae:.2f}</div>
    <br>
    <div class="metric-label">MAPE</div>
    <div style="font-size:22px;font-family:monospace;color:#f1c40f;">{mape:.1f}%</div>
    <br>
    <div class="metric-label">Horizon</div>
    <div style="font-size:22px;font-family:monospace;color:#f1c40f;">{forecast_days}d</div>
    <br>
    <hr style="border-color:#2a2a2a;">
    <small style="color:#666;">Evaluated on held-out 90-day test set. Lower MAPE = better fit.</small>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<p class="caption">Prophet uses additive/multiplicative decomposition with changepoints and custom quarterly seasonality. Confidence interval widens with forecast horizon.</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# FORECAST TABLE
# ─────────────────────────────────────────
with st.expander("📄 View Forecast Data Table"):
    future_only = forecast[forecast['ds'] > df['Date'].max()][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    future_only.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
    future_only = future_only.round(2)
    st.dataframe(future_only, use_container_width=True)

# ─────────────────────────────────────────
# EXECUTIVE SUMMARY
# ─────────────────────────────────────────
st.markdown('<div class="section-header">Executive Summary</div>', unsafe_allow_html=True)

rsi_latest = df['RSI'].iloc[-1]
macd_signal = "bullish" if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else "bearish"
bb_position = "upper band (overbought zone)" if df['Close'].iloc[-1] > df['BB_UPPER'].iloc[-1] else \
              "lower band (oversold zone)" if df['Close'].iloc[-1] < df['BB_LOWER'].iloc[-1] else \
              "middle of Bollinger Bands (neutral)"

st.markdown(f"""
<div class="insight-box">
<b>Trend:</b> The {df['MA30'].iloc[-1] > df['MA90'].iloc[-1] and "30-day MA is above the 90-day MA, suggesting a short-term bullish trend." or "30-day MA is below the 90-day MA, suggesting a short-term bearish trend."}<br><br>
<b>Momentum (RSI):</b> RSI is currently at <b>{rsi_latest:.1f}</b> — {'overbought territory, indicating potential near-term pullback.' if rsi_latest > 70 else 'oversold territory, indicating potential near-term recovery.' if rsi_latest < 30 else 'neutral territory with no extreme signal.'}<br><br>
<b>MACD:</b> Momentum is currently <b>{macd_signal}</b> based on MACD vs. signal line positioning.<br><br>
<b>Volatility:</b> Price is near the <b>{bb_position}</b>. Annualized volatility over the period is <b>{avg_vol:.1f}%</b>, which {'is elevated and signals higher risk.' if avg_vol > 40 else 'is within normal range for a large-cap growth stock.'}<br><br>
<b>Forecast:</b> The Prophet model projects a {forecast_days}-day outlook with a MAPE of {mape:.1f}%. The confidence interval reflects inherent uncertainty in longer horizons — treat point estimates as directional, not precise.<br><br>
<b>Investment Context:</b> Technical indicators should always be combined with fundamental analysis (earnings, subscriber growth, content spend) and macro conditions before drawing actionable conclusions.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<br><p style="font-size:11px;color:#444;font-family:monospace;">
⚠️ Disclaimer: This dashboard is for educational and analytical purposes only. Not financial advice.
</p>""", unsafe_allow_html=True)

