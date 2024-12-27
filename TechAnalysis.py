import TechCalculation as tc
import talib
import yfinance as yf


def analyze_RSI(symbol, start_date, end_date):
    RSI = tc.calculate_rsi(symbol, start_date, end_date)
    return RSI[-1] < 30


def analyze_SAR(symbol, start_date, end_date):
    close, SAR = tc.calculate_sar(symbol, start_date, end_date)
    return close > SAR


def analyze_SAR2(symbol, start_date, end_date):
    close, SAR = tc.calculate_sar2(symbol, start_date, end_date)
    return close[-2] < SAR[-2] and close[-1] > SAR[-1]


def analyze_STOCH(symbol, start_date, end_date, overbought_threshold=80, oversold_threshold=20):
    stochastic_data = tc.calculate_stochastic_oscillator(symbol, start_date, end_date)
    return stochastic_data['Stochastic_Oscillator'][-1] < oversold_threshold


def analyze_STOCH2(symbol, start_date, end_date):
    data = yf.download(symbol, start_date, end_date)
    stochastic_data = tc.calculate_stochastic_oscillator(symbol, start_date, end_date)

    if (data['Close'][-1] > data['Close'][-2]) and \
            (stochastic_data['Stochastic_Oscillator'][-2] < stochastic_data['Stochastic_Oscillator'][-2]):
        return True
    else:
        return False


def analyze_STOCH3(symbol, start_date, end_date):
    stochastic_data = tc.calculate_stochastic_oscillator(symbol, start_date, end_date)


    if stochastic_data['%K'][-1] > stochastic_data['%D'][-1] and stochastic_data['%K'][-2] <= stochastic_data['%D'][-2]:
        return True
    else:
        return False


def analyze_STOCH4(symbol, start_date, end_date):
    stochastic_data = tc.calculate_stochastic_oscillator(symbol, start_date, end_date)
    latest_row = stochastic_data.iloc[-1]
    if latest_row['%K'] > 20 and 20 < latest_row['%D'] < latest_row['%K']:
        return True
    else:
        return False


def analyze_OBV(symbol, start_date, end_date):
    obv_data = tc.calculate_obv(symbol, start_date, end_date)
    return obv_data[-1] > obv_data[-2]


def analyze_MACD(symbol, start_date, end_date):
    macd, signal = tc.calculate_macd(symbol, start_date, end_date)
    return macd[-1] > signal[-1]


def analyze_EMA(stock_symbol, start_date, end_date):
    ema_20 = tc.calculate_ema(stock_symbol, start_date, end_date)

    current_slope = (ema_20[-1] - ema_20[-2]) / ema_20[-2]
    previous_slope = (ema_20[-2] - ema_20[-3]) / ema_20[-3]

    return current_slope > 0 > previous_slope


def analyze_EMA_COMP(symbol, start_date, end_date):
    EMA = tc.calculate_ema(symbol, start_date, end_date)
    EMA2 = tc.calculate_ema_50(symbol, start_date, end_date)

    return EMA[-1] > EMA2[-1]


def analyze_BOLLINGER(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    current_price = stock_data['Close'][-1]

    bollinger_bands = tc.calculate_bollinger_bands(symbol, start_date, end_date)
    if bollinger_bands is None:
        return False

    upper_band, middle_band, lower_band = bollinger_bands
    lower_band = lower_band[-1]
    distance = (middle_band[-1] - lower_band) * 0.2

    return current_price < lower_band + distance


def analyze_ema_envelope(stock_symbol, start_date, end_date, envelope_percentage=5, area_percentage=1):
    ema = tc.calculate_ema(stock_symbol, start_date, end_date)

    lower_band = ema * (1 - envelope_percentage / 100)

    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    close_prices = stock_data['Close']

    last_close_price = close_prices.iloc[-1]

    return last_close_price <= lower_band[-1] * 1.01


def analyze_ema_envelope2(stock_symbol, start_date, end_date, ema_period=20, envelope_percentage=5):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

    close_prices = stock_data['Close']

    ema = talib.EMA(close_prices, timeperiod=ema_period)

    upper_band = ema * (1 + envelope_percentage / 100)

    last_close_price = close_prices[-1]

    return last_close_price < upper_band.iloc[-1] and close_prices[-2] >= upper_band.iloc[-2]


def analyze_EMA_alternative(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

    close_prices = stock_data['Close']

    ema_20 = talib.EMA(close_prices, timeperiod=20)

    last_close_price = close_prices[-1]
    last_ema = ema_20[-1]
    previous_close_price = close_prices[-2]

    return last_close_price > last_ema and previous_close_price < ema_20[-2]


def analyze_ADX_25(stock_symbol, start_date, end_date):
    adx = tc.calculate_adx(stock_symbol, start_date, end_date)
    return adx >= 25


def analyze_ADX_50(stock_symbol, start_date, end_date):
    adx = tc.calculate_adx(stock_symbol, start_date, end_date)
    return adx >= 50


def analyze_ADXR_25(stock_symbol, start_date, end_date):
    adxr_1 = tc.calculate_adx(stock_symbol, start_date, end_date)
    adxr_2 = tc.calculate_adxr2(stock_symbol, start_date, end_date)
    print(adxr_1)
    print(adxr_2)
    return adxr_2 < 25 < adxr_1


def analyze_ADXR_50(stock_symbol, start_date, end_date):
    adxr_1 = tc.calculate_adx(stock_symbol, start_date, end_date)
    adxr_2 = tc.calculate_adxr2(stock_symbol, start_date, end_date)
    return adxr_2 < 50 < adxr_1


def analyze_ADXR_CONF(stock_symbol, start_date, end_date):
    adxr_1 = tc.calculate_adx(stock_symbol, start_date, end_date)
    adxr_2 = tc.calculate_adxr2(stock_symbol, start_date, end_date)
    return adxr_1 > adxr_2


def analyze_ADXR_DIV(stock_symbol, start_date, end_date, adx_period=14, adxr_period=14):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

    high = stock_data['High']
    low = stock_data['Low']
    close = stock_data['Close']

    adx = talib.ADX(high, low, close, timeperiod=adx_period)
    adxr = (adx + adx.shift(adx_period)) / 2

    price = stock_data['Close']

    bullish_divergence = (price[-1] < price[-2]) & (adx[-1] > adx[-2]) & (adxr[-1] > adxr[-2])
    # bearish_divergence = (price[-1] > price[-2]) & (adx[-1] < adx[-2]) & (adxr[-1] < adxr[-2])

    return bullish_divergence


def analyze_APO(symbol, start_date, end_date):
    apo = tc.calculate_apo(symbol, start_date, end_date)
    return apo[-1] > 0 > apo[-2]


def analyze_APO_SIGNAL(symbol, start_date, end_date):
    apo = tc.calculate_apo(symbol, start_date, end_date)
    signal = tc.calculate_apo_signal_line(symbol, start_date, end_date)

    return apo[-1] > signal[-1] and apo[-2] < signal[-2]


def analyze_APO_SIGNAL2(symbol, start_date, end_date):
    apo = tc.calculate_apo(symbol, start_date, end_date)
    signal = tc.calculate_apo_signal_line(symbol, start_date, end_date)

    return apo[-1] > signal[-1]


def analyze_APO_DIV(symbol, start_date, end_date, fast_period=12, slow_period=26, matype=0):
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    close = stock_data['Close']
    apo = talib.APO(close, fastperiod=fast_period, slowperiod=slow_period, matype=matype)
    return (close[-1] < close[-2]) & (apo[-1] > apo[-2])


def analyze_APO_2(symbol, start_date, end_date):
    apo = tc.calculate_apo(symbol, start_date, end_date)
    return apo[-1] > apo[-2]

symbol = 'AAPL'
start_date = '2023-01-01'
end_date = '2024-03-16'

adx1 = analyze_ADXR_25(symbol, start_date, end_date)
print(adx1)
