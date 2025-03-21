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

data = get_data(ticker, start_date, end_date)

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

# 5. Baseline Model: Naive Forecast
# Predict the next value is the same as the current value
y_true = test_data['Close'].values
y_pred_naive = train_data['Close'].iloc[-1]  # Last value of training data, repeated for the test set
y_pred = [y_pred_naive] * len(test_data) # creates a list

# 6. Evaluate the Baseline Model
mae_naive = mean_absolute_error(y_true, y_pred)
rmse_naive = np.sqrt(mean_squared_error(y_true, y_pred))

st.write(f"### Naive Forecast - Baseline Model")
st.write(f"Mean Absolute Error (MAE): {mae_naive:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse_naive:.2f}")

# 7. Visualize the Predictions
st.write("### Naive Forecast vs. Actual Values")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(test_data['Close'], label='Actual', color='blue')
ax.axhline(y=y_pred_naive, color='red', linestyle='-', label=f'Naive Forecast: {y_pred_naive:.2f}') # show the forecast as a horizontal line
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.set_title('Naive Forecast vs. Actual Values')
ax.legend()
ax.grid(True)
st.pyplot(fig)
