import csv
import datetime
import json
import requests
import websocket
import yfinance as yf
from alpaca_trade_api.rest import REST, TimeFrame
from alpha_vantage.timeseries import TimeSeries
from pushbullet import Pushbullet
import PortfolioAugmentation2 as PA

day = 'Tuesday3'
end = "2024-04-16"
API_KEY = 'PKCV9EG5ZGJSMYZKDYLX'
API_SECRET = 'FMIKsSIRC3SoBvSpfifBiW0bCrJw2g9cuAVh7ZLP'
Alpaca_Second = 'PKUG5T2Y6ZOM2AUNNHP8'
Alpaca_Second_Secret = 'HeX5taJPkywzYb9bRbQRxdN1bXooeq5tmJHNE4Ki'
Alpaca_Third = 'PKANLCUG21XWBLH2INNA'
Alpaca_Third_Secret = 'RyN1qhihgNufKY8EFXlVEI9xyQGuvFqsmqTOjvm2'
OPENAI_API_KEY = "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"
ALPHA_VANTAGE_API_KEY = 'AIEDL2BYKOO73VLY'
PUSHBULLET_API_KEY = "o.inYcKT14jtjNfJdPJsJDG4vvh1rgh73H"

alpaca = REST(API_KEY, API_SECRET, base_url="https://paper-api.alpaca.markets", api_version='v2')
alpaca2 = REST(Alpaca_Second, Alpaca_Second_Secret, base_url="https://paper-api.alpaca.markets", api_version='v2')
alpaca3 = REST(Alpaca_Third, Alpaca_Third_Secret, base_url="https://paper-api.alpaca.markets", api_version='v2')

pb = Pushbullet(PUSHBULLET_API_KEY)

# Server < -- > Data Source
# Communication can go both ways
# Data source can send us information


wss = websocket.WebSocketApp("wss://stream.data.alpaca.markets/v1beta1/news")


def open_short_position(api, symbol, qty, order_type='market'):

    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side='sell',
            type=order_type,
            time_in_force='gtc'
        )
        return order._raw
    except Exception as e:
        print(f"Failed to open short position: {e}")
        return None


def check_bullish_engulfing(symbol):
    global end
    start_date = '2024-01-01'
    data = yf.download(symbol, start=start_date, end=end)

    if len(data) < 2:
        print("Insufficient data for analysis")
        return False

    yesterday = data.iloc[-2:]
    today = data.iloc[-1:]

    if today['Open'].iloc[0] < yesterday['Close'].iloc[0] < yesterday['Open'].iloc[0] < today['Close'].iloc[0] and \
       today['Close'].iloc[0] > today['Open'].iloc[0]:
        print("Bullish engulfing pattern found")
        return True
    else:
        print("No bullish engulfing pattern detected")
        return False


def on_open(ws):
    print("Websocket connected!")

    auth_msg = {
        "action": "auth",
        "key": API_KEY,
        "secret": API_SECRET
    }

    ws.send(json.dumps(auth_msg))

    subscribe_msg = {
        "action": "subscribe",
        "news": ["*"]  # ["TSLA"]
    }
    ws.send(json.dumps(subscribe_msg))


def get_stock_price(symbol):
    stock_data = yf.Ticker(symbol)

    try:
        current_price = stock_data.history(period='1d')['Close'].iloc[-1]
        return current_price

    except IndexError:
        print(f"Error: Unable to retrieve data for {symbol}")
        return None


def append(symbol, Headline, impact, filename):
    global end, day
    print("CHANGE!!!")
    stock_data = yf.download(symbol, start='2024-03-01', end=end)
    price_14 = stock_data['Close'][-14]
    price_10 = stock_data['Close'][-10]
    price_7 = stock_data['Close'][-7]
    price_5 = stock_data['Close'][-5]
    price_3 = stock_data['Close'][-3]
    price_1 = stock_data['Close'][-1]
    current = get_stock_price(symbol)

    row = [symbol, 'Alpaca', Headline, impact, price_14, price_10, price_7, price_5, price_3, price_1, current, '', '', '', '', '', '', day, 'OpenAI']

    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

    print(f"Added to {filename}")


def correction(symbol):
    global end
    print("CHANGE!!!")
    stock_data = yf.download(symbol, start='2024-03-01', end=end)
    current_price = get_stock_price(symbol)
    for i in range(1, 14):
        stock = stock_data['Close'][-i]
        print(stock)
        if stock > current_price * 1.025 or stock < current_price * 0.975:
            return False
    return True


def correction2(symbol):
    global end
    print("CHANGE!!!")
    stock_data = yf.download(symbol, start='2024-03-01', end=end)
    current_price = get_stock_price(symbol)
    for i in range(1, 10):
        stock = stock_data['Close'][-i]
        print(stock)
        if stock > current_price * 1.025 or stock < current_price * 0.975:
            return False
    return True


def on_message(ws, message):
    print("Message is " + message)
    current_event = json.loads(message)[0]
    if current_event["T"] == "n":
        company_impact = 0

        news_content = current_event['headline']

        api_request_body = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "Only respond with a number from 1-100 detailing the impact of the "
                                              "headline and body in a positive way to the future stock price."},
                {"role": "user", "content": f"Given the following news:\n{news_content}\nShow me a number from 1-100 "
                                            f"PREDICTING the positive impact on the FUTURE stock price."}
            ]
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": "Bearer " + OPENAI_API_KEY,
                "Content-Type": "application/json"
            },
            json=api_request_body
        ).json()

        print(response)
        print(response["choices"][0]["message"])
        company_impact = int(response["choices"][0]["message"]["content"])

        ticker_symbol = current_event["symbols"][0]
        append(ticker_symbol, news_content, company_impact, "All-Sentiment.csv")

        if company_impact >= 90:

            if correction(ticker_symbol):
                append(ticker_symbol, news_content, company_impact, "CorrectionAnalysis.csv")

            if correction2(ticker_symbol):
                append(ticker_symbol, news_content, company_impact, "CorrectionAnalysis-10.csv")

            append(ticker_symbol, news_content, company_impact, 'News-Sentiment-Analysis.csv')

            PA.order(ticker_symbol, API_KEY, API_SECRET)

            if check_bullish_engulfing(ticker_symbol):

                if correction(ticker_symbol):
                    append(ticker_symbol, news_content, company_impact, "Bullish-CorrectionAnalysis.csv")

                if correction2(ticker_symbol):
                    append(ticker_symbol, news_content, company_impact, "Bullish-CorrectionAnalysis-10.csv")

                append(ticker_symbol, news_content, company_impact, 'Bullish-Sentiment-Analysis.csv')

                pb.push_note("Bullish Engulfing Pattern Detected 90%",
                             f"Bullish Engulfing pattern detected for {ticker_symbol}")


        elif company_impact >= 80:

            if correction(ticker_symbol):
                append(ticker_symbol, news_content, company_impact, "CorrectionAnalysis.csv")

            if correction2(ticker_symbol):
                append(ticker_symbol, news_content, company_impact, "CorrectionAnalysis-10.csv")

            append(ticker_symbol,news_content, company_impact, 'News-Sentiment-Analysis.csv')

            PA.order(ticker_symbol, API_KEY, API_SECRET)

            if check_bullish_engulfing(ticker_symbol):

                if correction(ticker_symbol):
                    append(ticker_symbol, news_content, company_impact, "Bullish-CorrectionAnalysis.csv")

                if correction2(ticker_symbol):
                    append(ticker_symbol, news_content, company_impact, "Bullish-CorrectionAnalysis-10.csv")

                append(ticker_symbol, news_content, company_impact, 'Bullish-Sentiment-Analysis.csv')
                pb.push_note("Bullish Engulfing Pattern Detected 80%",
                             f"Bullish Engulfing pattern detected for {ticker_symbol}")

        elif company_impact >= 70:

            if correction(ticker_symbol):
                append(ticker_symbol, news_content, company_impact, "CorrectionAnalysis.csv")

            if correction2(ticker_symbol):
                append(ticker_symbol, news_content, company_impact, "CorrectionAnalysis-10.csv")

            append(ticker_symbol, news_content, company_impact, 'News-Sentiment-Analysis.csv')

            PA.order(ticker_symbol, API_KEY, API_SECRET)

            if check_bullish_engulfing(ticker_symbol):

                if correction(ticker_symbol):
                    append(ticker_symbol, news_content, company_impact, "Bullish-CorrectionAnalysis.csv")

                if correction2(ticker_symbol):
                    append(ticker_symbol, news_content, company_impact, "Bullish-CorrectionAnalysis-10.csv")

                append(ticker_symbol, news_content, company_impact, 'Bullish-Sentiment-Analysis.csv')
                pb.push_note("Bullish Engulfing Pattern Detected 70%",
                             f"Bullish Engulfing pattern detected for {ticker_symbol}")


        elif company_impact <= 50:
            append(ticker_symbol, news_content, company_impact, "Losing-Sentiment-Analysis.csv")

            current_price = get_stock_price(ticker_symbol)

            if current_price is not None:
                investment_amount = 1000
                quantity = investment_amount / current_price
                quantity = int(quantity)

                open_short_position(alpaca3, ticker_symbol, quantity)


wss.on_open = on_open
wss.on_message = on_message

wss.run_forever()
