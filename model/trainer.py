import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

def preprocess_for_lstm(df, feature='Close', window_size=60):
    """Scales and reshapes data into sequences for LSTM."""
    data = df[[feature]].values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i])
        y.append(scaled_data[i])  # Predict next value

    X, y = np.array(X), np.array(y)
    return X, y, scaler

def build_lstm_model(input_shape):
    """Builds and returns a compiled LSTM model."""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X, y, epochs=30, batch_size=32):
    """Train LSTM model with early stopping."""
    model = build_lstm_model((X.shape[1], X.shape[2]))
    es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[es])
    return model

def predict_future(model, data, window_size=60, n_days=5):
    """Predict next N days using the last input sequence."""
    predictions = []
    current_input = data[-1]  # shape = (window_size, features)

    for _ in range(n_days):
        current_input_reshaped = np.reshape(current_input, (1, window_size, 1))
        pred = model.predict(current_input_reshaped, verbose=0)
        predictions.append(pred[0][0])
        current_input = np.append(current_input[1:], [pred[0]], axis=0)

    return predictions