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

Total = pd.concat([News, Lose, Cor, Cor10, Bull, B_Cor, B_Cor10])
News.drop_duplicates(subset=['Headline'], inplace=True)
Total.drop_duplicates(subset=['Headline'], inplace=True)


def measure(file):

    p = 0
    n = 0
    up = []
    down = []

    for idx, row in file.iterrows():
        try:
            if row['Scored'] == 'LanModel' and row['Day'] == 'Friday':
                price = get_stock_price(row['Symbol'])
                if row['Current'] < price:
                    change = (price - row['Current']) / row['Current']

                    up.append(change)
                    p += 1
                else:

                    change = (price - row['Current']) / row['Current']
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


measure(News)

# 14.9, 'underweight', 'overweight', 'maintains', 'profit'
# 15.1, 'underweight', 'overweight', 'maintains', 'profit', 'closed'
# 18.9 'underweight', 'overweight', 'maintains', 'profit', 'closed'
# 20.7 'underweight', 'overweight', 'maintains', 'profit', 'closed', 'products'
# 21.7 'underweight', 'overweight', 'maintains', 'profit', 'closed', 'products', 'offer'
# words = ['underweight', 'overweight', 'maintains', 'profit', 'closed']
words = ['overweight', 'profit', 'closed', 'products', 'offering', 'soar']

H = News

News_filtered = H[~H['Headline'].str.lower().str.contains('|'.join(words))]
measure(News_filtered)



