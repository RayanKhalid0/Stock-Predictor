import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Fetch historical data
start_date = '2015-01-01'
end_date = '2024-01-01'
stock_symbol = 'GOOGL'  # You can change this to any stock symbol

data = yf.download(stock_symbol, start=start_date, end=end_date)

# Data Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

# Create training data
sequence_length = 60  # You can adjust this parameter
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
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_data, y_data, epochs=50, batch_size=32)

# Test data preparation
test_start_date = '2024-01-01'
test_end_date = '2024-05-01'
test_data = yf.download(stock_symbol, start=test_start_date, end=test_end_date)

actual_prices = test_data['Close'].values

total_data = pd.concat((data['Close'], test_data['Close']), axis=0)
model_inputs = total_data[len(total_data) - len(test_data) - sequence_length:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

x_test = []
for i in range(sequence_length, len(model_inputs)):
    x_test.append(model_inputs[i-sequence_length:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predictions
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plotting
plt.figure(figsize=(14,7))
#  plt.plot(test_data.index, actual_prices, color='black', label=f'Actual {stock_symbol} Price')
plt.plot( predicted_prices, color='green', label=f'Predicted {stock_symbol} Price')
plt.title(f'{stock_symbol} Price Prediction')
plt.xlabel('Date')
plt.ylabel(f'{stock_symbol} Price')
plt.legend()
plt.grid(True)
plt.show()
