import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Define the company symbol
company = 'TSLA'

# Define the time range for data retrieval
start = dt.datetime(2015, 1, 1)
end = dt.datetime(2024, 1, 1)

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

# Prepare data for prediction
latest_data = scaled_data[-prediction_days:]
latest_data = latest_data.reshape((1, prediction_days, 1))

# Predict stock prices for the next 60 days
predicted_prices_scaled = []

for _ in range(60):
    predicted_price = model.predict(latest_data)
    predicted_prices_scaled.append(predicted_price[0, 0])
    # Update the latest_data for the next prediction
    latest_data = np.roll(latest_data, -1)
    latest_data[0, -1, 0] = predicted_price

# Inverse transform the predicted prices
predicted_prices = scaler.inverse_transform(np.array(predicted_prices_scaled).reshape(-1, 1))

# Plot the results
plt.plot(predicted_prices, color='green', label=f"Predicted {company} price")
plt.title(f"{company} share price prediction")
plt.xlabel('Time (next 60 days)')
plt.ylabel(f'{company} Share price')
plt.legend()
plt.show()
