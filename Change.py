import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split

data = yf.download('NFTY', '2021-03-10', '2024-07-16')

data['Daily Change'] = data['Close'].pct_change().abs()

average_absolute_daily_percent_change = data['Daily Change'].mean() * 100
print(f"Average Absolute Daily Percent Change: {average_absolute_daily_percent_change}%")

data = data[int(len(data)*0.8):]
print(data.head())
