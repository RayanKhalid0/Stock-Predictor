import yfinance as yf
import talib
import requests
import matplotlib.pyplot as plt

api_key = 'pk_d4b12bb5420e436181069ebabb1e5bac'

symbols_url = f'https://cloud.iexapis.com/stable/ref-data/symbols?token={api_key}'
symbols_response = requests.get(symbols_url)
symbols_data = symbols_response.json()

for symbol_data in symbols_data:
    try:
        stock_symbol = symbol_data['symbol']
        stock_data = yf.download(stock_symbol, start='2023-01-01', end='2024-01-01')

        # Calculate SAR
        sar = talib.SAR(stock_data['High'], stock_data['Low'])
        latest_sar = sar.iloc[-1]
        latest_close_price = stock_data['Close'].iloc[-1]
        safety_distance_1 = 0.01 * latest_close_price
        safety_distance_2 = 0.02 * latest_close_price  # 2% below the stock price

        if (latest_close_price - safety_distance_2) < latest_sar < (
                latest_close_price - safety_distance_1):
            print(f"The Parabolic SAR for {stock_symbol} is between 1% and 2% below the stock price.")
            plt.figure(figsize=(14, 7))
            plt.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue')
            plt.plot(stock_data.index, sar, label='Parabolic SAR', color='red')
            plt.title(f'{stock_symbol} Close Price and Parabolic SAR')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print(f"The Parabolic SAR for {stock_symbol} is not between 1% and 2% below the stock price.")
    except Exception as e:
        print(f"Error processing {stock_symbol}: {str(e)}")
