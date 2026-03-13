import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

st.set_page_config(page_title="Netflix Stock Dashboard", layout="wide")

st.title("📈 Netflix Stock Analysis & Forecasting")
st.markdown("Dark‑mode dashboard with moving averages and Prophet forecasting.")

df = pd.read_csv("NFLX.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

df['MA30'] = df['Close'].rolling(30).mean()
df['MA90'] = df['Close'].rolling(90).mean()

plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df['Date'], df['Close'], color='#00FF66', label='Close Price')
ax.plot(df['Date'], df['MA30'], color='#00BFFF', label='30-Day MA')
ax.plot(df['Date'], df['MA90'], color='#1E90FF', label='90-Day MA')
ax.legend()
st.pyplot(fig)

prophet_df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
model = Prophet()
model.fit(prophet_df)

future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

fig2 = model.plot(forecast)
st.pyplot(fig2)

st.subheader("📄 Forecast Data")
st.dataframe(forecast.tail(20))
