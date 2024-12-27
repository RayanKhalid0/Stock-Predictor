import yfinance as yf
import pandas as pd

def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]  # Return the RSI value for today

# Define the ticker symbol
ticker_symbol = 'AAPL'  # Example: Apple Inc.

# Retrieve historical data from Yahoo Finance
data = yf.download(ticker_symbol, start='2024-02-01', end='2024-02-29')

# Calculate RSI for today
rsi_today = calculate_rsi(data)

print("RSI for today:", rsi_today)
