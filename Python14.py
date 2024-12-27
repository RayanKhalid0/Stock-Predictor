import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

company = 'AUTL'

# Historical data from 2018
start = dt.datetime(2018, 1, 1)
end = dt.datetime.now()

data = yf.download(company, start=start, end=end)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 30

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
model.fit(x_train, y_train, epochs=50, batch_size=32)

# Predict for each of the next 30 days
predictions = []
for i in range(120):
    real_data = scaled_data[-prediction_days:]
    real_data = np.expand_dims(real_data, axis=0)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    prediction = model.predict(real_data)
    predictions.append(prediction)
    # Update scaled_data to include the latest prediction for the next day
    scaled_data = np.append(scaled_data, prediction)

# Inverse transform to get actual prices
predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
next_120_days = pd.date_range(start=data.index[-1], periods=120)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Historical Prices (2018 onwards)')
plt.plot(next_120_days, predicted_prices, label='Predicted Prices (Next 120 days)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Prices Prediction')
plt.legend()
plt.grid(True)
plt.show()
