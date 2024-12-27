import re
import yfinance as yf
import pandas as pd

Words = [' under ', ' up ', '%', 'acquire', 'all-time low', 'announces', 'approve', 'bear', 'beat',
         'beat expectations', 'breakout', 'bullish', 'buy', 'buyback announcement', 'cut', 'decline',
         'dip', 'dismal results', 'down', 'drop', 'expand', 'expansion plan', 'fail', 'falls',
         'favorable news', 'feeling', 'from underperform to', 'growth opportunity', 'higher',
         'hold', 'increase', 'launch', 'layoff announcement', 'lower', 'maintain', 'merge', 'miss',
         'missed expectations', 'mixed', 'negative outlook', 'neutral', 'new short', 'nightmare',
         'outperform', 'overweight', 'plunge', 'positive outlook', 'potential sales dip',
         'profitable quarter', 'promising development', 'raise', 'raises price target', 'rally',
         'record high', 'record low', 'regulatory issues', 'repurchase', 'restructuring plan', 'rise',
         'sell-off', 'sliding', 'slip', 'slump', 'strong performance', 'surge', 'underperform',
         'underweight', 'unfavorable news', 'upward trend', 'weak', 'worse']

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


lower = ['lowers price target to']
raises = ['raises price target to']
pattern = r'\$(\d+)'

df = pd.DataFrame(columns=['Symbol'] + Words + ['Price_Target_Change', 'Success'])

for idx, row in Total.iterrows():
    try:
        row_data = {'Symbol': row['Symbol']}
        headline = row['Headline'].lower()

        for word in Words:
            row_data[word] = word in headline

        current_price = get_stock_price(row['Symbol'])
        for phrase_list, sign in zip([lower, raises], [-1, 1]):
            for phrase in phrase_list:
                if phrase in headline:
                    matches = re.findall(pattern, headline)
                    for match in matches:
                        price_target_change = sign * (float(match) - current_price) / current_price
                        row_data['Price_Target_Change'] = price_target_change

        success = (current_price - row['Current']) / row['Current']
        row_data['Success'] = success
        df = pd.concat([df, pd.DataFrame(row_data, index=[0])], ignore_index=True)
        print(df)
    except Exception as e:
        print(e)

df.to_csv('extracted_features.csv', index=False)
