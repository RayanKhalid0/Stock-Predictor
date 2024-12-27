import yfinance as yf
import talib
import matplotlib.pyplot as plt


stock_symbol = 'NVDA'
stock_data = yf.download(stock_symbol, start='2023-01-01', end='2024-07-10')

sar = talib.SAR(stock_data['High'], stock_data['Low'])

plt.figure(figsize=(14, 7))
plt.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue')
plt.plot(stock_data.index, sar, label='Parabolic SAR', color='red')
plt.title(f'{stock_symbol} Close Price and Parabolic SAR')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
