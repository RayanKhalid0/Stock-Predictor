import pandas as pd
import numpy as np
import yfinance as yf


def get_stock_price(ticker):
    stock_data = yf.Ticker(ticker)
    try:
        current = stock_data.history(period='1d')['Close'].iloc[-1]
        return current
    except IndexError:
        print(f"Error: {ticker}")
        return None


def get_p(symbol):
    data = yf.download(symbol, '2024-03-02', '2024-04-08')
    return data['Close'][-1]


News = pd.read_csv('News-Sentiment-Analysis.csv', encoding='latin-1')
Lose = pd.read_csv("Losing-Sentiment-Analysis.csv", encoding='latin-1')
All = pd.read_csv("All-Sentiment.csv", encoding='latin-1')
Lose.drop_duplicates(subset=['Headline', 'Day', 'Current', '1DAY'], inplace=True)
Bull = pd.read_csv('BullishAnalysis.csv')
p = 0
n = 0
up = []
down = []
high = ''
high2 = ''
h = -np.inf
h2 = np.inf
head = ''
head2 = ''
for idx, row in Lose.iterrows():
    try:
        if row['Day'] != '':

            price = get_stock_price(row['Symbol'])

            if row['Current'] < price and row['Symbol'] != 'WISA':
                change = (price - row['Current']) / row['Current']
                if change < 3:
                    if change > h:
                        h = change
                        high = row['Symbol']
                        head = row['Headline']
                        scored = row['Scored']

                    up.append(change)
                    p += 1
            elif row['Current'] > price:
                change = (price - row['Current']) / row['Current']
                if change < h2:
                    h2 = change
                    high2 = row['Symbol']

                    head2 = row['Headline']
                down.append(change)
                n += 1

    except Exception as e:
        print(e)

print(p)
print(n)
print(up)
print(down)
print(np.mean(up))
print(np.mean(down))

total = 1000

up_weighted_avg_return = np.average(up)
down_weighted_avg_return = np.average(down)

total_return = total * (
        1 + (len(up) * up_weighted_avg_return + len(down) * down_weighted_avg_return) / (len(down) + len(up)))

print(total_return)
print(((total_return - total) / total))

print(high)
print(h)
print(head)
print(scored)
print()
print(high2)
print(h2)
print(head2)
