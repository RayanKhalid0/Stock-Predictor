import numpy as np
import pandas as pd
import yfinance as yf


def get_stock_price(ticker):
    stock_data = yf.Ticker(ticker)
    try:
        current = stock_data.history(period='1d')['Close'].iloc[-1]
        return current
    except IndexError:
        print(f"Error: {ticker}")
        return None


Neutral = pd.read_csv('NeutralAnalysis.csv', encoding='latin-1')
changes = []
for idx, row in Neutral.iterrows():
    try:
        price = get_stock_price(row['Symbol'])
        current = row['Current']

        change = (price - current) / current
        changes.append(change)
    except Exception as e:
        print(e)

print(changes)
print(np.average(changes))
