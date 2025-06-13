from data.data_loader import fetch_stock_data # type: ignore
from utils.indicators import add_all_indicators # type: ignore
from model.trainer import preprocess_for_lstm, train_model, predict_future

import numpy as np

# Step 1: Fetch and process data
df = fetch_stock_data("AAPL", period="1y", interval="1d")
df = add_all_indicators(df)

# Step 2: Prepare LSTM input
X, y, scaler = preprocess_for_lstm(df, feature='Close', window_size=60)

# Step 3: Train the model
model = train_model(X, y)

# Step 4: Predict next 5 days
predictions = predict_future(model, X, n_days=5)

# Step 5: Invert scaling
predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Output
print("📈 Predicted Close Prices (Next 5 Days):")
for i, price in enumerate(predicted_prices.flatten(), 1):
    print(f"Day {i}: ${price:.2f}")