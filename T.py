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
Total.drop_duplicates(subset=['Headline'], inplace=True)

Days3 = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']


def measure(file, d):

    total = 1000

    print(d)
    p = 0
    n = 0
    up = []
    down = []

    for idx, row in file.iterrows():
        try:
            if row['Scored'] == 'LanModel' and row['Days'] == d:
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

    print("Day:", d)
    print("Positive:", p)
    print("Negative:", n)
    print("Up Changes:", up)
    print("Down Changes:", down)
    print("Average Up Change:", np.mean(up))
    print("Average Down Change:", np.mean(down))

    up_weighted_avg_return = np.average(up)
    down_weighted_avg_return = np.average(down)
    total_return = total * (1 + (len(up) * up_weighted_avg_return + len(down) * down_weighted_avg_return) / (len(down) + len(up)))

    print("Total Return:", total_return)
    print("Percentage Return:", ((total_return - total) / total) * 100)


for d in Days3:
    measure(Total, d)

words = ['underweight', 'overweight']
print("-----------------------------------------------------------")

News_filtered = Total[~Total['Headline'].str.lower().str.contains('|'.join(words))]
for d in Days3:
    measure(News_filtered, d)
