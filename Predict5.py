import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf  # Yahoo Finance library
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Define the company symbol
company = 'AAPl'

# Define the time range for data retrieval
start = dt.datetime(2015, 1, 1)
end = dt.datetime(2024, 1, 1)  # Extend the end date for training

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

# Prepare test data for the next 60 days after 2024-01-01
test_start = dt.datetime(2024, 1, 1)
test_end = dt.datetime(2024, 3, 1)  # Predicting the next 60 days

future_dates = pd.date_range(start=test_start, end=test_end)

# Make predictions for the next 60 days
x_future = scaled_data[-prediction_days:]
x_future = np.reshape(x_future, (1, prediction_days, 1))

predicted_prices = []

for _ in range(60):
    prediction = model.predict(x_future)
    predicted_prices.append(prediction)
    x_future = np.append(x_future[:,1:,:], prediction.reshape(1, 1, 1), axis=1)  # Reshaping prediction before appending

predicted_prices = np.array(predicted_prices)
predicted_prices = np.reshape(predicted_prices, (predicted_prices.shape[0], 1))

# Inverse scaling
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the results
plt.plot(data.index, data['Close'], color="black", label=f"Actual {company} price")
plt.plot(future_dates, predicted_prices, color='green', label=f"Predicted {company} price")
plt.title(f"{company} share price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share price')
plt.legend()
plt.show()