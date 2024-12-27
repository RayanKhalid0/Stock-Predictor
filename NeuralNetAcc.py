import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt
from concurrent.futures import ThreadPoolExecutor

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

def predict_prices(model, x_test, scaler):
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    return predicted_prices

company = 'NVDA'

start = dt.datetime(2015, 1, 1)
end = dt.datetime.now()


def calculate(symbol):
    data = yf.download(company, start=start, end=end)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    prediction_days = 120

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=10, batch_size=32)

    test_start = dt.datetime(2023, 1, 1)
    test_end = dt.datetime.now()

    test_data = yf.download(company, start=test_start, end=test_end)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']))

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x-prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))



    with ThreadPoolExecutor(max_workers=10) as executor:
        predictions = [executor.submit(predict_prices, model, x_test, scaler) for _ in range(10)]

    predicted_prices_array = [future.result() for future in predictions]

    average_predicted_prices = np.mean(predicted_prices_array, axis=0)

    real_data = model_inputs[-prediction_days:]
    real_data = np.expand_dims(real_data, axis=0)
    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f"Average prediction for the next day: {np.mean(prediction)}")