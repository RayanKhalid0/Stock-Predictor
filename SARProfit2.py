import yfinance as yf
import talib
import matplotlib.pyplot as plt

# Fetch historical stock data
stock_symbol = 'BBLGW'  # You can change this to any stock symbol
stock_data = yf.download(stock_symbol, start='2023-01-01', end='2024-01-01')

sar = talib.SAR(stock_data['High'], stock_data['Low'])

buy_price = 0
total_profit = 0
buy_dates = []
sell_dates = []

# Iterate through each date
for i in range(1, len(stock_data)):
    current_sar = sar.iloc[i]
    current_close_price = stock_data['Close'].iloc[i]

    # Check if SAR crossed the stock price from above to below (potential buy signal)
    if sar.iloc[i - 1] > stock_data['Close'].iloc[i - 1] and current_sar <= current_close_price:
        if buy_price == 0:
            buy_price = current_close_price
            buy_dates.append(stock_data.index[i])
            print(f"Buy Signal: {stock_data.index[i].date()} - Buy Price: {buy_price}")
        else:
            profit = current_close_price - buy_price
            total_profit += profit
            print(f"Sell Signal: {stock_data.index[i].date()} - Sell Price: {current_close_price} - Profit: {profit:.2f}")
            sell_dates.append(stock_data.index[i])
            buy_price = 0

    # Check if stock increases by 1% (potential sell signal)
    elif buy_price != 0 and current_close_price >= buy_price * 1.15:
        profit = current_close_price - buy_price
        total_profit += profit
        print(f"Sell Signal: {stock_data.index[i].date()} - Sell Price: {current_close_price} - Profit: {profit:.2f}")
        sell_dates.append(stock_data.index[i])
        buy_price = 0

    # Check if SAR crossed back up (potential sell signal)
    elif sar.iloc[i - 1] > current_sar and sar.iloc[i - 2] <= sar.iloc[i - 1] and buy_price != 0:
        profit = current_close_price - buy_price
        total_profit += profit
        print(f"Sell Signal (SAR): {stock_data.index[i].date()} - Sell Price: {current_close_price} - Profit: {profit:.2f}")
        sell_dates.append(stock_data.index[i])
        buy_price = 0

# Calculate percentage change compared to the first buy
percentage_change = (total_profit / stock_data['Close'].iloc[0]) * 100

print(f"Total Numeric Profit: {total_profit:.2f}")
print(f"Percentage Change compared to the first buy: {percentage_change:.2f}%")

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue')
plt.plot(stock_data.index, sar, label='Parabolic SAR', color='red')

# Plot buy signals
for buy_date in buy_dates:
    plt.axvline(x=buy_date, color='green', linestyle='--', linewidth=1)

# Plot sell signals
for sell_date in sell_dates:
    plt.axvline(x=sell_date, color='purple', linestyle='--', linewidth=1)

plt.title(f'{stock_symbol} Close Price and Parabolic SAR with Buy and Sell Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
