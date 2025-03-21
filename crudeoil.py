import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Data Acquisition
@st.cache_data  # Cache the data to avoid repeated downloads
def get_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

ticker = "CL=F"  # WTI Crude Oil Futures
start_date = "2010-01-01"
end_date = "2024-01-01"

data = get_data(ticker, start_date, end_date)

# 2. Data Cleaning: Check for Missing Values
st.write("### Missing values before handling:")
st.write(data.isnull().sum())

# Handle Missing Values (Forward Fill)
data.fillna(method='ffill', inplace=True)

st.write("### Missing values after handling:")
st.write(data.isnull().sum())

# 3. Feature Engineering: Lagged Features
# Create lagged features for the closing price
data['Close_Lag1'] = data['Close'].shift(1)
data['Close_Lag2'] = data['Close'].shift(2)
data['Close_Lag3'] = data['Close'].shift(3)
data['Close_Lag7'] = data['Close'].shift(7)  # One week lag
data['Close_Lag30'] = data['Close'].shift(30)  # One month lag

# Drop any rows with remaining NaN values (due to the shift operation creating NaNs at the beginning)
data.dropna(inplace=True)

# Display the first few rows with the new features
st.write("### Data with lagged features:")
st.dataframe(data.head())  # Use st.dataframe for displaying DataFrames

# Visualize the lagged features
st.write("### Crude Oil Price with Lagged Features")
fig, ax = plt.subplots(figsize=(12, 6))  # Create a matplotlib figure and axes
ax.plot(data['Close'], label='Close')
ax.plot(data['Close_Lag1'], label='Close Lag 1')
ax.plot(data['Close_Lag7'], label='Close Lag 7')
ax.set_title('Crude Oil Price with Lagged Features')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend()
ax.grid(True)
st.pyplot(fig)  # Display the matplotlib figure in Streamlit
