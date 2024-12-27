import talib
import yfinance as yf


def calculate_obv(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    obv = talib.OBV(stock_data['Close'], stock_data['Volume'])

    return obv


def calculate_sar(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    close = stock_data["Close"][-1]
    sar = talib.SAR(stock_data['High'], stock_data['Low'])

    return close, sar[-1]


def calculate_sar2(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    close = stock_data["Close"]
    sar = talib.SAR(stock_data['High'], stock_data['Low'])

    return close, sar


def calculate_stochastic_oscillator(symbol, start_date, end_date, period=14, smooth_k=3, smooth_d=3):
    data = yf.download(symbol, start=start_date, end=end_date)

    data['Lowest_Low'] = data['Low'].rolling(window=period).min()
    data['Highest_High'] = data['High'].rolling(window=period).max()
    data['%K'] = ((data['Close'] - data['Lowest_Low']) / (data['Highest_High'] - data['Lowest_Low'])) * 100

    data['%D'] = data['%K'].rolling(window=smooth_k).mean()

    data['Stochastic_Oscillator'] = data['%D'].rolling(window=smooth_d).mean()

    return data[['%K', '%D', 'Stochastic_Oscillator']]


def calculate_bollinger_bands(ticker_symbol, start_date, end_date, window=20, num_std=2):
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

    close_prices = stock_data['Close'].values

    upper_band, middle_band, lower_band = talib.BBANDS(close_prices, timeperiod=window, nbdevup=num_std,
                                                       nbdevdn=num_std)

    return upper_band, middle_band, lower_band


def calculate_ema(stock_symbol, start_date, end_date, period=20):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    close_prices = stock_data['Close']
    ema = talib.EMA(close_prices, timeperiod=period)

    return ema


def calculate_ema_50(stock_symbol, start_date, end_date, period=50):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    close_prices = stock_data['Close']
    ema = talib.EMA(close_prices, timeperiod=period)

    return ema


def calculate_rsi(symbol, start_date, end_date, period=14):
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    close_prices = stock_data['Close'].values

    rsi = talib.RSI(close_prices, timeperiod=period)

    return rsi


def calculate_macd(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    macd, signal, _ = talib.MACD(stock_data['Close'])

    return macd, signal


def calculate_adx(symbol, start_date, end_date, period=14):
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    high = stock_data['High']
    low = stock_data['Low']
    close = stock_data['Close']

    adx = talib.ADX(high, low, close, timeperiod=period)

    return adx[-1]


def calculate_adxr(symbol, start_date, end_date, period=14):
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    high = stock_data['High']
    low = stock_data['Low']
    close = stock_data['Close']

    adx = talib.ADX(high, low, close, timeperiod=period)

    adxr = (adx + adx.shift(period)) / 2
    adxr = adxr.dropna()
    adxr1 = float(adxr[-1])
    if len(adxr) >= 2:
        return adxr1
    else:
        return None, None


def calculate_adxr2(symbol, start_date, end_date, period=14):
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    high = stock_data['High']
    low = stock_data['Low']
    close = stock_data['Close']

    adx = talib.ADX(high, low, close, timeperiod=period)

    adxr = (adx + adx.shift(period)) / 2
    adxr = adxr.dropna()

    if len(adxr) >= 2:
        print("B")
        return adxr[-2]
    else:
        return None, None


def calculate_apo(symbol, start_date, end_date, fast_period=12, slow_period=26, matype=0):
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    close = stock_data['Close']

    apo = talib.APO(close, fastperiod=fast_period, slowperiod=slow_period, matype=matype)

    return apo


def calculate_apo_signal_line(symbol, start_date, end_date, fast_period=12, slow_period=26, signal_period=9, matype=0):
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    close = stock_data['Close']

    apo = talib.APO(close, fastperiod=fast_period, slowperiod=slow_period, matype=matype)

    signal_line = talib.SMA(apo, timeperiod=signal_period)

    return signal_line


symbol = 'AAPL'
start_date = '2023-01-01'
end_date = '2024-03-16'


