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

# Adding technical indicators such as moving averages
data['MA10'] = data['Close'].rolling(window=10).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()

# Drop rows with missing values resulting from calculating moving averages
data.dropna(inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close', 'MA10', 'MA50']])

prediction_days = 30

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, :])  # Include features Close, MA10, MA50
    y_train.append(scaled_data[x, 0])  # Predicting only Close price

x_train, y_train = np.array(x_train), np.array(y_train)

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=2, batch_size=32)

predictions = []
for i in range(30):
    real_data = scaled_data[-prediction_days:]
    real_data = np.expand_dims(real_data, axis=0)
    prediction = model.predict(real_data)
    predictions.append(prediction[0])  # Extracting the scalar prediction value
    # Update scaled_data to include the latest prediction for the next day
    # Only update the Close price, not the additional features like MA10 and MA50
    scaled_data = np.append(scaled_data, [prediction], axis=0)

# Inverse transform to get actual prices
predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Historical Prices (2018 onwards)')
plt.plot(pd.date_range(start=data.index[-1], periods=30), predicted_prices, label='Predicted Prices (Next 30 days)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Prices Prediction')
plt.legend()
plt.grid(True)
plt.show()
