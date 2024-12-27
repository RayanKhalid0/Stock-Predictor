import pandas as pd
import requests

data = pd.read_csv("DATA-METRICS.csv")

token = "pk_4a9546b33fc6433db703cf2ca47b7dff"

industries = []
symbols_url = f'https://cloud.iexapis.com/stable/ref-data/symbols?token={token}'
symbols_response = requests.get(symbols_url)
symbols_data = symbols_response.json()
count = 0
for symbol_data in symbols_data:
    symbol = symbol_data['symbol']
    count += 1
    print(count)
    print(symbol)

print(count)
