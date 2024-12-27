from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import yfinance as yf


def get_stock_price(ticker):
    stock_data = yf.Ticker(ticker)
    try:
        current = stock_data.history(period='1d')['Close'].iloc[-1]
        return current
    except IndexError:
        print(f"Error: Unable to retrieve data for {ticker}")
        return None


headlines = []
scores = []

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
News.drop_duplicates(subset=['Headline', 'Day', 'Current', '1DAY'], inplace=True)


for idx, row in News.iterrows():
    if row['Day'] == 'Tuesday':
        try:
            price = get_stock_price(row['Symbol'])
            current = row['Current']
            change = (price - current) / current
            scores.append(change)
            headlines.append(row['Headline'])

        except Exception as e:
            print(e)

scores = np.array(scores)
len(scores)
len(headlines)
vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = vectorizer.fit_transform(headlines)

feature_names = vectorizer.get_feature_names_out()

tfidf_array = tfidf_matrix.toarray()

weighted_scores = np.dot(tfidf_array.T, scores)

positive_indices = weighted_scores >= 0
negative_indices = weighted_scores < 0

sorted_positive_indices = np.argsort(weighted_scores[positive_indices])[::-1]
sorted_negative_indices = np.argsort(weighted_scores[negative_indices])

sorted_positive_features = [feature_names[i] for i in sorted_positive_indices]
sorted_negative_features = [feature_names[i] for i in sorted_negative_indices]

num_keywords = 10
top_positive_keywords = sorted_positive_features[:num_keywords]
print("Top positive keywords contributing to the score:")
for keyword in top_positive_keywords:
    print(keyword)

top_negative_keywords = sorted_negative_features[:num_keywords]
print("\nTop negative keywords contributing to the score:")
for keyword in top_negative_keywords:
    print(keyword)

while True:
    continue