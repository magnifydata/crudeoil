# 1. Install Libraries (if you don't have them already)
# pip install yfinance pandas numpy matplotlib scikit-learn statsmodels

# 2. Import Libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 3. Data Acquisition
ticker = "CL=F"  # WTI Crude Oil Futures
start_date = "2010-01-01"
end_date = "2024-01-01"

data = yf.download(ticker, start=start_date, end=end_date)

# 4. Data Exploration
print(data.head())
print(data.tail())
data.info()

# 5. Plot the Closing Price
plt.figure(figsize=(12, 6))
plt.plot(data['Close'])
plt.title('Crude Oil Price (WTI)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.show()
