import streamlit as st
import pandas as pd
import numpy as np
from data.data_loader import fetch_stock_data  # type: ignore
from utils.indicators import add_all_indicators # type: ignore
from model.trainer import preprocess_for_lstm, train_model, predict_future

import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# --- Sidebar Inputs ---
st.sidebar.title("📈 Stock Price Prediction")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", min_value=1, max_value=30, value=5)

if st.sidebar.button("🔄 Refresh & Predict"):
    with st.spinner("Fetching data and making predictions..."):

        # Step 1: Get data
        df = fetch_stock_data(ticker)
        df = add_all_indicators(df)

        # Step 2: Prepare & train model
        X, y, scaler = preprocess_for_lstm(df, feature='Close', window_size=60)
        model = train_model(X, y)
        predictions = predict_future(model, X, n_days=forecast_days)
        predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

        # Step 3: Append predictions to timeline
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date, periods=forecast_days+1, freq='B')[1:]
        df_pred = pd.DataFrame({'Date': future_dates, 'Predicted_Close': predicted_prices}).set_index('Date')

        # Step 4: Display chart
        st.subheader(f"📊 Predicted Close Prices for {ticker}")
        fig, ax = plt.subplots(figsize=(12, 6))
        df['Close'].plot(ax=ax, label="Historical", color='skyblue')
        df_pred['Predicted_Close'].plot(ax=ax, label="Forecast", color='orange')
        ax.set_title(f"{ticker} Price Prediction")
        ax.legend()
        st.pyplot(fig)

        # Step 5: Show data
        st.subheader("📉 Technical Indicators")
        st.dataframe(df.tail(10).style.format("{:.2f}"))

        st.subheader("🧠 Prediction Output")
        st.dataframe(df_pred.style.format("₹{:.2f}"))

else:
    st.warning("👈 Enter a stock ticker and click 'Refresh & Predict'")