import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Data Acquisition (same as before, with caching)
@st.cache_data
def get_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

ticker = "CL=F"  # WTI Crude Oil Futures
start_date = "2010-01-01"
end_date = "2024-01-01"

data = get_data(ticker, start=start_date, end=end_date)

# 2. Data Cleaning (same as before)
data.fillna(method='ffill', inplace=True)

# 3. Feature Engineering (same as before)
data['Close_Lag1'] = data['Close'].shift(1)
data['Close_Lag2'] = data['Close'].shift(2)
data['Close_Lag3'] = data['Close'].shift(3)
data['Close_Lag7'] = data['Close'].shift(7)
data['Close_Lag30'] = data['Close'].shift(30)
data.dropna(inplace=True)

# 4. Split Data into Training and Testing Sets
test_size = 0.2  # 20% for testing
test_index = int(len(data) * (1 - test_size))

train_data = data[:test_index]
test_data = data[test_index:]

st.write(f"Training data from: {train_data.index.min()} to {train_data.index.max()}")
st.write(f"Testing data from: {test_data.index.min()} to {test_data.index.max()}")

# 5. Baseline Model: Naive Forecast (same as before)
y_true = test_data['Close'].values
y_pred_naive = test_data['Close'].shift(1).fillna(train_data['Close'].iloc[-1]).values

mae_naive = mean_absolute_error(y_true, y_pred_naive)
rmse_naive = np.sqrt(mean_squared_error(y_true, y_pred_naive))

st.write(f"### Naive Forecast - Baseline Model")
st.write(f"Mean Absolute Error (MAE): {mae_naive:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse_naive:.2f}")

# 6. Simple Moving Average (SMA) Model
window_size = st.slider("SMA Window Size", min_value=3, max_value=60, value=7)  # Add a slider for window size

# Calculate moving average on the training data
train_data['SMA'] = train_data['Close'].rolling(window=window_size).mean()

# Fill NaN values in the beginning of the SMA with the mean of the 'Close' column in the training data
train_data['SMA'].fillna(train_data['Close'].mean(), inplace=True)

# Make predictions using the SMA on the test data
y_pred_sma = []
for i in range(len(test_data)):
    if i < window_size - 1: #If the index is less than window size just take the mean of the close price for the training dataset
        y_pred_sma.append(train_data['Close'].mean())
    else:
        y_pred_sma.append(train_data['Close'][-window_size:].mean())

# Evaluate the SMA Model
mae_sma = mean_absolute_error(y_true, y_pred_sma)
rmse_sma = np.sqrt(mean_squared_error(y_true, y_pred_sma))

st.write(f"### Simple Moving Average (SMA) - Window Size: {window_size}")
st.write(f"Mean Absolute Error (MAE): {mae_sma:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse_sma:.2f}")

# 7. Visualize the Predictions (SMA and Naive Forecast)
st.write("### SMA Forecast vs. Actual Values")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(test_data['Close'], label='Actual', color='blue')
ax.plot(test_data.index, y_pred_naive, label='Naive Forecast', color='green')
ax.plot(test_data.index, y_pred_sma, label=f'SMA ({window_size} days)', color='red')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.set_title('SMA Forecast vs. Actual Values')
ax.legend()
ax.grid(True)
st.pyplot(fig)
