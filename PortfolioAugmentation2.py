import requests
import yfinance as yf


def get_stock_price(symbol):
    stock_data = yf.Ticker(symbol)

    try:
        current_price = stock_data.history(period='1d')['Close'].iloc[-1]
        return current_price

    except IndexError:
        print(f"Error in getting stock price: {symbol}")
        return None


def order(symbol, key, secret):
    try:
        total_investment = 1000000
        url = "https://paper-api.alpaca.markets/v2/orders"
        url2 = f"https://paper-api.alpaca.markets/v2/assets/{symbol}"

        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret
        }

        response = requests.get(url, headers=headers).json()
        IDS = {}
        for i in response:
            IDS[i['id']] = get_stock_price(i['symbol'])

        response2 = requests.get(url2, headers=headers).json()
        IDS[response2['id']] = get_stock_price(response2['symbol'])
        print(IDS)
        investment = total_investment / len(IDS)
        print(investment)

        for key, val in IDS.items():
            url3 = f"https://paper-api.alpaca.markets/v2/orders/{key}"
            qty = investment / val
            payload = {"qty": qty}
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "APCA-API-KEY-ID": key,
                "APCA-API-SECRET-KEY": secret
            }

            response = requests.patch(url3, json=payload, headers=headers)
            print(response)
    except Exception as e:
        print(f"Error in replacing Orders {e}")




