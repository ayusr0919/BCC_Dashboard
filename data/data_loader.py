import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# def fetch_stock_data(ticker="AAPL", start="2020-01-01", end="2025-12-31", interval="1d"):
#     """Fetch historical stock data with no adjustments."""
#     df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False)
#     df.dropna(inplace=True)
#     return df


def fetch_stock_data(ticker="AAPL", period="1y", interval="1d"):
    """Fetch historical stock data."""
    df = yf.download(ticker, period=period, interval=interval)
    df.dropna(inplace=True)
    return df

def scale_data(df, feature_cols):
    """Scale selected features using MinMaxScaler."""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])
    scaled_df = pd.DataFrame(scaled_data, columns=feature_cols, index=df.index)
    return scaled_df, scaler

def prepare_lstm_sequences(data, window_size=60):
    """
    Prepare sequences and labels for LSTM.
    Returns: X, y as numpy arrays.
    """
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i][0])  # Assuming first column = 'Close'
    return np.array(X), np.array(y)

def add_technical_indicators(df, indicators_func):
    """
    Adds tech indicators to df using a passed function (from indicators.py)
    """
    return indicators_func(df)