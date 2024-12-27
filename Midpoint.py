import yfinance as yf
import talib
import matplotlib.pyplot as plt


def calculate_midpoint(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    close_prices = stock_data['Close']

    midpoint = talib.MIDPOINT(close_prices, timeperiod=14)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(close_prices)
    plt.plot(midpoint, label='Midpoint', color='blue')
    plt.title('Midpoint of Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Midpoint')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
if __name__ == "__main__":
    symbol = 'BBLGW'  # Example stock symbol (Apple Inc.)
    start_date = '2023-01-01'  # Start date for fetching data
    end_date = '2024-01-01'  # End date for fetching data

    calculate_midpoint(symbol, start_date, end_date)
