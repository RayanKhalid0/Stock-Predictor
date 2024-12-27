import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

def get_data(company, start, end):
    return yf.download(company, start=start, end=end)

def preprocess_data(data, prediction_days):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    x, y = [], []
    for i in range(prediction_days, len(scaled_data)):
        x.append(scaled_data[i - prediction_days:i, 0])
        y.append(scaled_data[i, 0])

    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x, y, scaler

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_predictions(actual_prices, predicted_prices):
    plt.plot(actual_prices, color='black', label='Actual Prices')
    plt.plot(predicted_prices, color='green', label='Predicted Prices')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def main():
    company = 'GOOGL'
    start_train = dt.datetime(2015, 1, 1)
    end_train = dt.datetime(2023, 1, 1)
    end_test = dt.datetime.now()
    prediction_days = 60

    data_train = get_data(company, start_train, end_train)
    data_test = get_data(company, end_train, end_test)

    x_train, y_train, scaler = preprocess_data(data_train['Close'], prediction_days)
    x_test, _, _ = preprocess_data(data_test['Close'], prediction_days)

    model = build_model((x_train.shape[1], 1))

    model.fit(x_train, y_train, epochs=50, batch_size=32)

    predictions = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predictions)

    actual_prices = data_test['Close'].values
    plot_predictions(actual_prices, predicted_prices)

if __name__ == "__main__":
    main()
