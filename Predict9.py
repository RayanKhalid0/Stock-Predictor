import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Fetch historical data
start_date = '2010-01-01'  # Extend the training data period
end_date = '2023-12-31'      # Extend the training data period
stock_symbol = 'AAPL'

data = yf.download(stock_symbol, start=start_date, end=end_date)

# Data Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Create training data
sequence_length = 60
x_data = []
y_data = []

for i in range(sequence_length, len(scaled_data)):
    x_data.append(scaled_data[i-sequence_length:i, 0])
    y_data.append(scaled_data[i, 0])

x_data, y_data = np.array(x_data), np.array(y_data)
x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_data.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_data, y_data, epochs=50, batch_size=32)

# Test data preparation
test_start_date = '2024-01-01'
test_end_date = '2024-03-01'
test_data = yf.download(stock_symbol, start=test_start_date, end=test_end_date)

# Prepare test data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_test_data = scaler.fit_transform(test_data['Close'].values.reshape(-1, 1))

x_test = []
for i in range(sequence_length, len(scaled_test_data)):
    x_test.append(scaled_test_data[i-sequence_length:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predictions for the next 30 days
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(test_data.index, test_data['Close'], color='blue', label='Actual AAPL Price')
plt.plot(test_data.index[sequence_length:], predicted_prices, color='green', label='Predicted AAPL Price')
plt.title('AAPL Price Prediction')
plt.xlabel('Date')
plt.ylabel('AAPL Price')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
