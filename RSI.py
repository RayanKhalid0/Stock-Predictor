import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

ticker_symbol = 'BBLGW'
data = yf.download(ticker_symbol, start='2023-02-03', end='2024-03-01')
data2 = yf.download(ticker_symbol, start='2024-02-14', end='2024-03-22')

# Calculate RSI
data['RSI'] = calculate_rsi(data)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Close Price', color='blue')
plt.title('Stock Price and RSI')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc='upper left')

print(calculate_rsi(data2))

# Create a secondary y-axis for RSI
ax2 = plt.gca().twinx()

# Plot RSI and highlight points where RSI < 30
ax2.plot(data.index, data['RSI'], label='RSI', color='red')
ax2.fill_between(data.index, y1=0, y2=30, where=(data['RSI'] < 30), color='green', alpha=0.3)
ax2.set_ylabel('RSI')
ax2.legend(loc='upper right')

plt.show()
