import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Conv1D
from keras.src.optimizers import Adam
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import yfinance as yf

data = yf.download('NFTY', start='2008-01-01', end='2024-07-16')
data.fillna(method='ffill', inplace=True)

data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()
data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
data['RSI_14'] = 100 - (
            100 / (1 + data['Close'].diff().apply(lambda x: np.where(x > 0, x, 0)).rolling(window=14).mean() /
                   data['Close'].diff().apply(lambda x: np.where(x < 0, -x, 0)).rolling(window=14).mean()))
data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
data['Bollinger_Upper'] = data['Close'].rolling(window=20).mean() + (data['Close'].rolling(window=20).std() * 2)
data['Bollinger_Lower'] = data['Close'].rolling(window=20).mean() - (data['Close'].rolling(window=20).std() * 2)
data.fillna(method='bfill', inplace=True)
data['Close_Target'] = data['Close'].shift(-1)

features = ['Open', 'High', 'Close', 'Low', 'Volume', 'SMA_50', 'SMA_200']
target = 'Close_Target'

data.drop(columns=['Adj Close'], axis=1, inplace=True)
data.dropna(inplace=True)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features + [target]])

X = scaled_data[:, :-1]
y = scaled_data[:, -1]

X = X.reshape(X.shape[0], X.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    Conv1D(filters=32, kernel_size=2, activation='relu'),
    Dropout(0.2),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.0008), loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=1000, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=200, batch_size=56, validation_data=(X_test, y_test), shuffle=False, callbacks=[early_stopping])
model.save('CNN-NFTY.h5')

loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

predictions = model.predict(X_test)

predicted_prices = scaler.inverse_transform(np.concatenate((X_test.reshape(X_test.shape[0], X_test.shape[1]), predictions), axis=1))[:, -1]
actual_prices = scaler.inverse_transform(np.concatenate((X_test.reshape(X_test.shape[0], X_test.shape[1]), y_test.reshape(-1, 1)), axis=1))[:, -1]



mse = mean_squared_error(actual_prices, predicted_prices)
mape = mean_absolute_percentage_error(actual_prices, predicted_prices)
r2 = r2_score(actual_prices, predicted_prices)
mae = mean_absolute_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)

print(f"Mean squared Error - MSE: {mse}")
print(f"Mean Absolute Percentage Error - MAPE: {mape}")
print(f"R-squared R^2: {r2}")
print(f"Mean Absolute Error - MAE: {mae}")
print(f"Root Mean Square Error - RMSE: {rmse}")

count_true = 0
count_false = 0

for idx in range(1, len(predicted_prices)):
    if (predicted_prices[idx] < actual_prices[idx - 1] and actual_prices[idx] < actual_prices[idx - 1]) or \
            (predicted_prices[idx] > actual_prices[idx - 1] and actual_prices[idx] > actual_prices[idx - 1]):
        count_true += 1
    else:
        count_false += 1

print(f"False Trends: {count_false}")
print(f"True Trends: {count_true}")

accuracy = np.mean([1 - abs(actual_prices[i] - predicted_prices[i]) / actual_prices[i] for i in range(len(predicted_prices))]) * 100
print(f"Accuracy: {accuracy}")

plt.figure(figsize=(14, 7))
plt.plot(data.index[-len(y_test):], actual_prices, color='blue', label='Actual Values')
plt.plot(data.index[-len(y_test):], predicted_prices, color='red', label='Predicted Values')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Actual vs Predicted Closing Prices CNN')
plt.legend()
plt.grid(True)
plt.show()

