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

T = [-0.30860417676246543, -0.09513325661549253, -0.07500000186264512, -0.04933527725610359, -0.04772945933890939, -0.04695651842200238, -0.04132231274684204, -0.04105090209131846, -0.03781512408618731, -0.03463681257528833, -0.021546271124294407, -0.018518610500991368, -0.01487206520337452, -0.011425567341356888, -0.01134214427451975, -0.009836584562160744, -0.006741803303391998, -0.004836905350592607, -0.004274526301752855, -0.0008405374180772621, 0.0, 0.001175805849179969, 0.003596050559435024, 0.004107720365800046, 0.0050834928377325814, 0.005395721244081881, 0.006763609678620385, 0.010737417466434305, 0.014234523000662441, 0.018214380620433734, 0.025957977031656327, 0.02774643562894399, 0.02870365374699957, 0.03871316690078409, 0.056115158852643454, 0.06741573409952314, 0.07802543865410438, 0.08272134216673004, 0.097717481280618, 0.10910590017744289, 0.10956093493421212, 0.14081149034143042, 0.14487635932292492, 0.15506198597931703, 0.18976899442674927, 0.2915601227312962]

word = 'trading higher'
changes = []
for i in T:

    for idx, row in Total.iterrows():
        try:
            price = get_stock_price(row['Symbol'])
            current = row['Current']
            change = (price - current) / current
            if change == i:
                print(f"{row['Headline']} : {change}")
        except Exception as e:
            print(e)

T = sorted(changes)
print(T)
print(np.average(changes))


