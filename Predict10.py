import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Fetch historical data
start_date = '2015-01-01'
end_date = '2023-12-31'
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
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_data, y_data, epochs=50, batch_size=32)

# Predictions for the next 30 days beyond the last available data point
future_days = 30
last_data_point = scaled_data[-sequence_length:]
predicted_prices_scaled = []

for _ in range(future_days):
    x_test = np.reshape(last_data_point, (1, sequence_length, 1))
    predicted_price_scaled = model.predict(x_test)
    predicted_prices_scaled.append(predicted_price_scaled[0,0])
    last_data_point = np.append(last_data_point[1:], predicted_price_scaled, axis=0)

predicted_prices = scaler.inverse_transform(np.array(predicted_prices_scaled).reshape(-1,1))

# Generate future dates for plotting
last_date = pd.to_datetime(data.index[-1])
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='D')

plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], color='blue', label='Actual AAPL Price')
plt.plot(future_dates, predicted_prices, color='green', label='Predicted AAPL Price')
plt.title('AAPL Price Prediction')
plt.xlabel('Date')
plt.ylabel('AAPL Price')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
