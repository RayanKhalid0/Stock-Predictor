import requests

symbol = 'AAPL'
url = f"https://data.alpaca.markets/v1beta1/options/snapshots/{symbol}"

headers = {
    "accept": "application/json",
    "APCA-API-KEY-ID": "PKUG5T2Y6ZOM2AUNNHP8",
    "APCA-API-SECRET-KEY": "HeX5taJPkywzYb9bRbQRxdN1bXooeq5tmJHNE4Ki"
}

response = requests.get(url, headers=headers)
print(response.json())

from alpaca_trade_api import REST

Alpaca_Second = 'PKUG5T2Y6ZOM2AUNNHP8'
Alpaca_Second_Secret = 'HeX5taJPkywzYb9bRbQRxdN1bXooeq5tmJHNE4Ki'

alpaca2 = REST(Alpaca_Second, Alpaca_Second_Secret, base_url="https://paper-api.alpaca.markets", api_version='v2')

order1 = {
    "symbol": "AAPL240503P00100000",
    "qty": 1,
    "side": "buy",
    "type": "market",
    "time_in_force": "day"
}

try:
    alpaca2.submit_order(
        symbol=order1['symbol'],
        qty=order1['qty'],
        side=order1['side'],
        type=order1['type'],
        time_in_force=order1['time_in_force']
    )
    print("Option order placed successfully.")
except Exception as e:
    print("Error placing option order:", str(e))
