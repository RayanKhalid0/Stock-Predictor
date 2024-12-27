import yfinance as yf
import talib
import matplotlib.pyplot as plt
import numpy as np

# Define the stock symbol and the timeframe
stock_symbol = 'AAPL'
start_date = '2023-01-01'
end_date = '2024-01-01'

# Fetch historical stock price data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Convert the 'Close' prices to a NumPy array of type double
close_prices = np.array(stock_data['Close'], dtype=np.double)

# Remove NaN values
close_prices = close_prices[~np.isnan(close_prices)]

# Calculate MAVP using talib
mavp_period = 10.0  # You can adjust this period as needed
periods = np.arange(5.0, 15.0)  # Convert range to NumPy array

print("Length of close_prices:", len(close_prices))
print("Length of periods:", len(periods))

# Make sure the lengths of close_prices and periods match
if len(close_prices) >= max(periods):
    mavp = talib.MAVP(close_prices, periods=periods, minperiod=2, maxperiod=30, matype=0)  # Adjust parameters as needed
else:
    print("Error: Not enough data points to compute MAVP with the given periods.")

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(stock_data.index[:len(close_prices)], close_prices, label='Close Price', color='blue')
for i in range(len(mavp)):
    plt.plot(stock_data.index[len(close_prices)-len(mavp[i]):len(close_prices)], mavp[i], label=f'MAVP {periods[i]}', alpha=0.7)
plt.title('MAVP with Variable Periods')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
