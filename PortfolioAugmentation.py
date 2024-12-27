import requests
import yfinance as yf


def get_stock_price(symbol):
    stock_data = yf.Ticker(symbol)

    try:
        current_price = stock_data.history(period='1d')['Close'].iloc[-1]
        return current_price

    except IndexError:
        print(f"Error: Unable to retrieve data for {symbol}")
        return None


def submit_order(api, symbol, news_content, investment_amount):
    current_price = get_stock_price(symbol)

    if current_price is not None:
        quantity = investment_amount / current_price
        quantity = int(quantity)

        order = api.submit_order(
            symbol=symbol,
            qty=quantity,
            side='buy',
            type='market',
            time_in_force='day'
        )
        print(f"Bought {quantity} shares of {symbol} at ${current_price} per share.")
        with open("trades_log.txt", "a") as f:
            f.write(f"Bought {quantity} shares of {symbol} at ${current_price} per share.\n")
            f.write(f"News: {news_content}\n")
            f.write("\n")
        return order


def order(api, symbol, news_content, key, secret):
    url = "https://paper-api.alpaca.markets/v2/positions"

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret
    }

    total = 1000000
    response = requests.get(url, headers=headers).json()
    print(response)
    symbols = []
    for i in response:
        symbols.append(i['symbol'])

    symbols.append(symbol)
    print(symbols)
    investment_amount = total / (len(symbols))
    print(investment_amount)

    requests.delete(url, headers=headers)

    for sym in symbols:
        submit_order(api, sym, news_content, investment_amount)


