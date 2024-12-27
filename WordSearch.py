import pandas as pd
import numpy as np
import yfinance as yf

News = pd.read_csv('News-Sentiment-Analysis.csv', encoding='latin-1')
Lose = pd.read_csv('Losing-Sentiment-Analysis.csv', encoding='latin-1')
Cor10 = pd.read_csv('CorrectionAnalysis-10.csv')
Cor = pd.read_csv('CorrectionAnalysis.csv')
Bull = pd.read_csv('Bullish-Sentiment-Analysis.csv')
B_Cor = pd.read_csv('Bullish-CorrectionAnalysis.csv')
B_Cor10 = pd.read_csv('Bullish-CorrectionAnalysis-10.csv')

Total = pd.concat([News, Lose, Cor, Cor10, Bull, B_Cor, B_Cor10])
Total.drop_duplicates(subset=['Headline', 'Day', 'Current', '1DAY'], inplace=True)


def get_stock_price(ticker):
    stock_data = yf.Ticker(ticker)
    try:
        current = stock_data.history(period='1d')['Close'].iloc[-1]
        return current
    except IndexError:
        print(f"Error: {ticker}")
        return None


high = -np.inf
low = np.inf
h1 = ''
h2 = ''
s1 = ''
s2 = ''
# legal
# risk
# ~
# merge

# good
# not so good
# bad
# potential
# raises
# buy
word = ['surprise']
changes = []
for idx, row in News.iterrows():
    try:

        day = row['Current']
        headline = row['Headline']
        headline = headline.lower()
        for w in word:
            if w in headline:
                current = get_stock_price(row['Symbol'])
                change = (current - day) / day
                changes.append(change)
                print(row)
                print(headline)
                print(change)
                print()

    except Exception as e:
        print(e)

print(changes)
print(np.average(changes))
