import pandas as pd
import numpy as np
import yfinance as yf


def get_stock_price(ticker):
    stock_data = yf.Ticker(ticker)
    try:
        current = stock_data.history(period='1d')['Close'].iloc[-1]
        return current
    except IndexError:
        print(f"Error: Unable to retrieve data for {ticker}")
        return None


News = pd.read_csv('News-Sentiment-Analysis.csv', encoding='latin-1')
Lose = pd.read_csv('Losing-Sentiment-Analysis.csv', encoding='latin-1')
Cor10 = pd.read_csv('CorrectionAnalysis-10.csv')
Cor = pd.read_csv('CorrectionAnalysis.csv')
Bull = pd.read_csv('Bullish-Sentiment-Analysis.csv')
B_Cor = pd.read_csv('Bullish-CorrectionAnalysis.csv')
B_Cor10 = pd.read_csv('Bullish-CorrectionAnalysis-10.csv')

Total = pd.concat([News, Lose, Cor, Cor10, Bull, B_Cor, B_Cor10])

Total.reset_index(drop=True, inplace=True)
Total.drop_duplicates(inplace=True)

change = []
pos = 0
neg = 0
high = -np.inf
low = np.inf
Symbol = ""

for idx, row in Lose.iterrows():
    if row['Day'] == 'Monday':
        symbol = row['Symbol']
        impact = row['Impact']
        if impact >= 0:
            price = get_stock_price(symbol)
            if price is not None:
                current_price = row['Current']
                if current_price < price:
                    pos += 1
                else:
                    neg += 1
                perc = (price - current_price) / current_price
                if perc > high:
                    high = perc
                    Symbol = symbol
                if perc < low:
                    low = perc

                change.append(perc)

change = [value for value in change if not np.isnan(value)]
av = np.mean(change)

print(f"The number of positive: {pos}")
print(f"The number of negative: {neg}")
print(f"The change across: {change}")
print(f"The average change across: {av}")
print(f"The highest: {high}, {Symbol}")
print(f"The lowest: {low}")
