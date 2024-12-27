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

words = ['opening', 'appointments', 'closes', 'merging', 'surge', 'partnerships', 'open', 'closed', 'falls', 'gains', 'compete', 'plunging', 'report', 'increases', 'promotions', 'agreements', 'slides', 'reveal', 'announced', 'expanding', 'sell', 'growth', 'restructured', 'announcing', 'layoffs', 'weakening', 'agreement', 'sliding', 'increase', 'soars', 'releasing', 'ventures', 'divest', 'slide', 'launch', 'launched', 'downsize', 'decline', 'reveals', 'divesting', 'growing', 'strategy', 'strengthening', 'gainful', 'weakens', 'bids', 'unveiled', 'buys', 'selling', 'bid', 'offers', 'expands', 'partnership', 'appointment', 'increasing', 'declines', 'job', 'plunge', 'merges', 'launching', 'divestiture', 'appoints', 'layoff', 'gain', 'revealing', 'grows', 'close', 'results', 'profit', 'strengthen', 'rise', 'rising', 'weak', 'unveils', 'agreed', 'release', 'reporting', 'acquiring', 'promotion', 'competing', 'sales', 'venture', 'lost', 'offer', 'collaborate', 'promote', 'earnings', 'estimate', 'collaborates', 'expansions', 'cut', 'launches', 'purchasing', 'restructures', 'agreeing', 'promotes', 'jointly', 'downsizes', 'merge', 'acquisitions', 'plunges', 'promoting', 'competitiveness', 'declined', 'hitting', 'downsizing', 'appoint', 'hiring', 'opened', 'forecast', 'hire', 'grow', 'collaborating', 'profits', 'soar', 'strong', 'hit', 'acquires', 'cutting', 'falling', 'appointing', 'partnering', 'strategic', 'guidance', 'jump', 'venturing', 'strengthens', 'released', 'acquisition', 'revenues', 'announcement', 'losing', 'restructuring', 'projection', 'opens', 'hits', 'fall', 'partner', 'reported', 'laying', 'expansion', 'rises', 'expanded', 'unveil', 'declining', 'unveiling', 'hired', 'revealed', 'hires', 'closing', 'buy', 'jumps', 'purchases', 'acquire', 'profitable', 'deal', 'sells', 'competitive', 'joint', 'product', 'announces', 'products', 'collaboration', 'divestitures', 'agree', 'divests', 'losses', 'jobs', 'announce', 'offering', 'buying', 'surges', 'releases', 'cuts', 'purchase', 'decreasing', 'competitor', 'sale', 'revenue', 'quarterly', 'expand', 'loss', 'strength', 'decreases', 'soaring', 'merger', 'dividend', 'strategically', 'reports', 'bidding', 'restructure', 'competition', 'weaken', 'decrease', 'surging', 'jumping', 'financial', 'downsized', 'outlook', 'annual']


def get_stock_price(ticker):
    stock_data = yf.Ticker(ticker)
    try:
        current = stock_data.history(period='1d')['Close'].iloc[-1]
        return current
    except IndexError:
        print(f"Error: {ticker}")
        return None


changes = []

word_dict = {word: 0 for word in words}
dict2 = {word: 0 for word in words}

for idx, row in Total.iterrows():
    try:
        day = row['Current']
        headline = row['Headline'].lower()
        current = get_stock_price(row['Symbol'])
        if current is not None:
            change = (current - day) / day
            for w in words:
                if w in headline:
                    word_dict[w] += change
                    dict2[w] += 1
                    print(f"{w} : {change}")
    except Exception as e:
        print(f"Error processing row {idx}: {e}")

print(word_dict)

changes = [change for change in changes if change is not None]
average_change = np.average(changes)
print("Average Change:", average_change)

dict3 = {word: 0.0 for word in words}


for key, value in word_dict.items():
    try:

        av = value/(dict2[key])
        dict3[key] = av
    except ZeroDivisionError as e:
        print()

print(word_dict)
print(dict2)
print(dict3)
