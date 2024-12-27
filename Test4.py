import pandas as pd
import yfinance as yf
import numpy as np


def get_stock_price(ticker):
    stock_data = yf.Ticker(ticker)
    try:
        current = stock_data.history(period='1d')['Close'].iloc[-1]
        return current
    except IndexError:
        print(f"Error: {ticker}")
        return None


News = pd.read_csv('News-Sentiment-Analysis.csv', encoding='latin-1')
Lose = pd.read_csv('Losing-Sentiment-Analysis.csv', encoding='latin-1')
Cor10 = pd.read_csv('CorrectionAnalysis-10.csv')
Cor = pd.read_csv('CorrectionAnalysis.csv')
Bull = pd.read_csv('Bullish-Sentiment-Analysis.csv')
B_Cor = pd.read_csv('Bullish-CorrectionAnalysis.csv')
B_Cor10 = pd.read_csv('Bullish-CorrectionAnalysis-10.csv')
Neutral = pd.read_csv('NeutralAnalysis.csv', encoding='latin-1')

Total = pd.concat([News, Lose, Cor, Cor10, Bull, B_Cor, B_Cor10, Neutral])
Total.drop_duplicates(subset=['Headline', 'Day', 'Current', '1DAY'], inplace=True)

word = 'trading lower'
headline_changes = []

for idx, row in Total.iterrows():
    if row['Day'] == 'Wednesday' and row['Scored'] == 'LanModel':
        headline = row['Headline'].lower()
        if word in headline:
            price = get_stock_price(row['Symbol'])
            current = row['Current']
            change = (price - current) / current

            headline_changes.append({'headline': headline, 'change': change})

sorted_headlines = sorted(headline_changes, key=lambda x: x['change'])

for entry in sorted_headlines:
    print(entry['headline'], entry['change'])
