import yfinance as yf
import talib
import matplotlib.pyplot as plt

ticker_symbol = 'TSLA'
start_date = '2023-01-01'
end_date = '2024-03-08'

stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
# MAMA line crosses above the FAMA line, indicating a potential uptrend
# Helps in divergence as it can point out potential reversals
mama, fama = talib.MAMA(stock_data['Close'])

plt.figure(figsize=(12, 6))
plt.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue')
plt.plot(stock_data.index, mama, label='MAMA', color='red')
plt.plot(stock_data.index, fama, label='FAMA', color='green')

plt.title(f'{ticker_symbol} Stock Price and MAMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
