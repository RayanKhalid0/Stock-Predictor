import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf  # Yahoo Finance library
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Define the company symbol
company = 'AAPL'

# Define the time range for data retrieval
start = dt.datetime(2015, 1, 1)
end = dt.datetime.now()

# Retrieve data from Yahoo Finance
data = yf.download(company, start=start, end=end)

# Extract the 'Close' prices
close_prices = data['Close'].values.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Define the number of days for prediction
prediction_days = 60

# Prepare the training data
x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=25, batch_size=32)

# Prepare test data
test_data = yf.download(company, start=end, end=end + dt.timedelta(days=60))

# Extract the 'Close' prices for test data
actual_prices = test_data['Close'].values

# Scale the test data
scaled_test_data = scaler.transform(actual_prices.reshape(-1, 1))

x_test = []

for x in range(prediction_days, len(scaled_test_data)):
    x_test.append(scaled_test_data[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predict stock prices
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the results
plt.plot(actual_prices, color="black", label=f"Actual {company} price")
plt.plot(predicted_prices, color='green', label=f"Predicted {company} price")
plt.title(f"{company} share price prediction for the next 60 days")
plt.xlabel('Time')
plt.ylabel(f'{company} Share price')
plt.legend()
plt.show()
