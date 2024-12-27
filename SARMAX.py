import yfinance as yf
import talib
import matplotlib.pyplot as plt
import requests

api_key = 'pk_d4b12bb5420e436181069ebabb1e5bac'

MAX = 0
MIN = 1000
Symbol_Min = ""
Symbol_Max = ""
TotalN = 0
TotalP = 0


def analyze_stock_signals(stock_symbol, start_date, end_date):
    global MAX, MIN, Symbol_Min, Symbol_Max, TotalN, TotalP

    try:
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        stock_data.fillna(method='ffill', inplace=True)

        sar = talib.SAR(stock_data['High'], stock_data['Low'])

        buy_price = 0
        total_profit = 0
        buy_dates = []
        sell_dates = []

        First = True
        first_buy = 0

        for i in range(1, len(stock_data)):
            current_sar = sar.iloc[i]
            current_close_price = stock_data['Close'].iloc[i]

            if sar.iloc[i - 1] > stock_data['Close'].iloc[i - 1] and current_sar <= current_close_price:
                if buy_price == 0:
                    buy_price = current_close_price
                    buy_dates.append(stock_data.index[i])
                    if First:
                        first_buy = buy_price
                    print(f"Buy Signal: {stock_data.index[i].date()} - Buy Price: {buy_price}")
                else:
                    profit = current_close_price - buy_price
                    total_profit += profit
                    print(f"Sell Signal: {stock_data.index[i].date()} - Sell Price: {current_close_price} - Profit: {profit:.2f}")
                    sell_dates.append(stock_data.index[i])
                    buy_price = 0

            elif sar.iloc[i - 1] < stock_data['Close'].iloc[i - 1] and current_sar >= current_close_price:
                if buy_price != 0:
                    profit = current_close_price - buy_price
                    total_profit += profit
                    print(f"Sell Signal: {stock_data.index[i].date()} - Sell Price: {current_close_price} - Profit: {profit:.2f}")
                    sell_dates.append(stock_data.index[i])
                    buy_price = 0

        if len(stock_data['Close']) > 0:
            percentage_change = (total_profit / first_buy) * 100
        else:
            percentage_change = 0

        print(f"Total Numeric Profit: {total_profit:.2f}")
        print(f"Percentage Change compared to the first buy: {percentage_change:.2f}%")

        if percentage_change > MAX:
            MAX = percentage_change
            Symbol_Max = stock_symbol

        if percentage_change < MIN:
            MIN = percentage_change
            Symbol_Min = stock_symbol

        if percentage_change > 0:
            TotalP = TotalP + 1
        else:
            TotalN = TotalN + 1


    except Exception as e:
        print(f"Error analyzing {stock_symbol}: {str(e)}")


symbols_url = f'https://cloud.iexapis.com/stable/ref-data/symbols?token={api_key}'
symbols_response = requests.get(symbols_url)
symbols_data = symbols_response.json()

for symbol_data in symbols_data:
    ticker_symbol = symbol_data['symbol']
    analyze_stock_signals(ticker_symbol, '2023-01-01', '2024-03-09')

print(f"The maximum percentage profit was {MAX} with symbol {Symbol_Max}")
print(f"The minimum percentage profit was {MIN} with symbol {Symbol_Min}")

print(TotalP)
print(TotalN)
