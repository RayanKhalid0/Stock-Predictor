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


def get_stock_price(ticker):
    stock_data = yf.Ticker(ticker)
    try:
        current = stock_data.history(period='1d')['Close'].iloc[-1]
        return current
    except IndexError:
        print(f"Error: {ticker}")
        return None
# Total = Total.drop(['Scored'], axis=1)


Total.drop_duplicates(subset=['Headline', 'Day', 'Current', '1DAY'], inplace=True)

word1 = ['trading lower']
word2 = ['trading higher']

UP_lower = 0
UP_higher = 0
D_lower = 0
D_higher = 0

above = []
below = []
Headline1 = []
Headline2 = []
Headline3 = []
Headline4 = []


for idx, row in Total.iterrows():
    headline = row['Headline']
    headline = headline.lower()
    if row['Scored'] == 'LanModel' and row['Day'] == 'Tuesday':
        price = get_stock_price(row['Symbol'])
        for word in word1:
            if word in headline:
                if row['Current'] < price:
                    Headline1.append(headline)
                    UP_lower += 1
                    change = (price - row['Current']) / row['Current']
                    above.append(change)
                else:
                    Headline3.append(headline)
                    D_lower += 1
                    change = (price - row['Current']) / row['Current']
                    below.append(change)

        for word in word2:
            if word in headline:
                if row['Current'] < price:
                    UP_higher += 1
                    change = (price - row['Current']) / row['Current']
                    above.append(change)
                    Headline2.append(headline)

                else:
                    Headline4.append(headline)

                    D_higher += 1
                    change = (price - row['Current']) / row['Current']
                    below.append(change)


av1 = np.mean(above)
av2 = np.mean(below)

print(f"Above with trading lower: {UP_lower}")
print(f"Below with trading lower: {D_lower}")
print(f"Above with trading higher: {UP_higher}")
print(f"Below with trading higher: {D_higher}")
print(f"Change for Above: {above}")
print(f"Change for below: {below}")
print(f"Average change for Above: {av1}")
print(f"Average change for below: {av2}")
print("\nHeadlines for lowers with higher: IMPOSTER")
print(Headline1)
print("\nHeadlines for raises with higher:")
print(Headline2)
print("\nHeadlines for lowers with lower:")
print(Headline3)
print("\nHeadlines for raises with lower: IMPOSTER")
print(Headline4)



