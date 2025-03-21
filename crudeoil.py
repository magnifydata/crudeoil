import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime
#from statsmodels.tsa.ar.model import AutoReg #removed and adding the line below
from statsmodels.tsa.arima.model import ARIMA

# 1. Data Acquisition (same as before, with caching)
@st.cache_data
def get_data(ticker: str, start: datetime, end: datetime): # Change end_date to end
    try:
        data = yf.download(ticker, start=start, end=end) #Change end_date to end
        return data
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None

ticker = "CL=F"  # WTI Crude Oil Futures
start_date = datetime.datetime(2010, 1, 1) #Added datetime object
end_date = datetime.datetime(2024, 1, 1) #Added datetime object

data = get_data(ticker, start_date, end_date)

if data is None:
    st.stop()  # Stop execution if data download fails

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

# Calculate moving average on the training data - WE DONT NEED IT ANYMORE
#train_data['SMA'] = train_data['Close'].rolling(window=window_size).mean()

# Fill NaN values in the beginning of the SMA with the mean of the 'Close' column in the training data - WE DONT NEED IT ANYMORE
#train_data['SMA'].fillna(train_data['Close'].mean(), inplace=True)

# Make predictions using the SMA on the test data
y_pred_sma = []
# Combine train and test data for rolling window calculation
combined_data = pd.concat([train_data['Close'], test_data['Close']], axis=0)

# Calculate SMA on the combined data
sma_values = combined_data.rolling(window=window_size).mean()

# Extract SMA predictions for the test period
y_pred_sma = sma_values[train_data.index[-1]:].iloc[1:].values

# Evaluate the SMA Model
mae_sma = mean_absolute_error(y_true, y_pred_sma)
rmse_sma = np.sqrt(mean_squared_error(y_true, y_pred_sma))

st.write(f"### Simple Moving Average (SMA) - Window Size: {window_size}")
st.write(f"Mean Absolute Error (MAE): {mae_sma:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse_sma:.2f}")

# 7. Autoregression (AR) Model
# Initialize list to store predictions
y_pred_ar = []

# Define the ARIMA model order
order = (5, 0, 0)

# Use all data for training
history = combined_data['Close'].values.tolist()  #  changed
# Iterate through the test data and make predictions
for i in range(len(test_data)):
    # Train the AR model
    model = ARIMA(history[:len(train_data) + i], order=order)
    model_fit = model.fit()
    # Make predictions on the test data
    output = model_fit.forecast()
    yhat = output[0]
    # Append the prediction to the list
    y_pred_ar.append(yhat)

    # update history to what is used to be known.
    history.append(test_data['Close'].iloc[i]) # append the testing value.

# Evaluate the AR Model
mae_ar = mean_absolute_error(y_true, y_pred_ar)
rmse_ar = np.sqrt(mean_squared_error(y_true, y_pred_ar))

st.write(f"### Autoregression (AR) Model")
st.write(f"Mean Absolute Error (MAE): {mae_ar:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse_ar:.2f}")

# 8. Visualize the Predictions (All Models)
st.write("### Forecast Comparison")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(test_data['Close'], label='Actual', color='blue')
ax.plot(test_data.index, y_pred_naive, label='Naive Forecast', color='green')
ax.plot(test_data.index, y_pred_sma, label=f'SMA ({window_size} days)', color='red')
ax.plot(test_data.index, y_pred_ar, label='AR Model', color='orange')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.set_title('Forecast Comparison')
ax.legend()
ax.grid(True)
st.pyplot(fig)
