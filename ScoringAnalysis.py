import pandas as pd
import yfinance as yf
import LanguageAnalysis as LA

def get_stock_price(ticker):
    stock_data = yf.Ticker(ticker)
    try:
        current = stock_data.history(period='1d')['Close'].iloc[-1]
        return current
    except IndexError:
        print(f"Error: {ticker}")
        return None


News = pd.read_csv("News-Sentiment-Analysis.csv")
UP = {}
Low = {}
for idx, row in News.iterrows():
    try:
        if row['Scored'] == 'LanModel' and row['Day'] == 'Thursday':
            headline = row['Headline'].lower()
            if 'raises price target' in headline:
                current = row['Current']
                price = get_stock_price(row['Symbol'])
                change = (price - row['Current']) / row['Current']
                score1 = LA.score(headline, current)
                if change > 0:
                    UP.update({change: score1})
                else:
                    Low.update({change: score1})
    except Exception as e:
        print(e)

print(UP)
print(Low)

t = 0
p = 0
for key, value in Low.items():
    t += 1
    if value == 0:
        p += 1

print(t)
print(p)








