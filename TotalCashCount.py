import requests


token = "pk_4a9546b33fc6433db703cf2ca47b7dff"

def totalCash(symbol):
    global token
    url = f"https://api.iex.cloud/v1/data/core/advanced_stats/{symbol}?token={token}"
    response2 = (requests.get(url)).json()
    print(response2)
    for item in response2:
        for key, value in item.items():
            if key == 'totalCash':
                if value is not None:
                    return value
                else:
                    return 0


symbols_url = f'https://cloud.iexapis.com/stable/ref-data/symbols?token={token}'
symbols_response = requests.get(symbols_url)
symbols_data = symbols_response.json()
count = 0
none = 0
some = 0

for symbol_data in symbols_data:
    symbol = symbol_data['symbol']
    if totalCash(symbol) == 0:
        none += 1
    else:
        some += 1
    count += 1

print(f"Count: {count}")
print(f"None values: {none}")
print(f"Not none values: {some}")


print(totalCash("NVDA"))
