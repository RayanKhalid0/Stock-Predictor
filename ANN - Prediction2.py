import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.callbacks import EarlyStopping
from keras.src.optimizers import Adam
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import yfinance as yf

data = yf.download('NFTY', start='2008-01-01', end='2024-07-16')
data.ffill(inplace=True)
data.bfill(inplace=True)


data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()
data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
data['RSI_14'] = 100 - (100 / (1 + data['Close'].diff().apply(lambda x: np.where(x > 0, x, 0)).rolling(window=14).mean() /
                                data['Close'].diff().apply(lambda x: np.where(x < 0, -x, 0)).rolling(window=14).mean()))
data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
data['Bollinger_Upper'] = data['Close'].rolling(window=20).mean() + (data['Close'].rolling(window=20).std() * 2)
data['Bollinger_Lower'] = data['Close'].rolling(window=20).mean() - (data['Close'].rolling(window=20).std() * 2)



data['Target'] = data['Close'].shift(-1)
data.dropna(inplace=True)

features = ['Open', 'High', 'Close', 'Low', 'Volume', 'SMA_50', 'SMA_200']


scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(data[features])

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(data['Target'].values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

model = Sequential([
    Dense(units=128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(units=64, activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=16, activation='relu'),
    Dense(units=8, activation='relu'),
    Dense(units=1)
])


model.compile(optimizer=Adam(learning_rate=0.005), loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=1000, restore_best_weights=True)
history = model.fit(X_train, y_train, batch_size=64, epochs=1000, validation_data=(X_test, y_test), callbacks=[early_stopping])

y_pred = model.predict(X_test)
y_test_inv = scaler_y.inverse_transform(y_test)
y_pred_inv = scaler_y.inverse_transform(y_pred)

mse = mean_squared_error(y_test_inv, y_pred_inv)
mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)
mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mse)

print(f"Mean Squared Error on the test set: {mse}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")
print(f"R-squared (R²) on the test set: {r2}")
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Square Error: {rmse}")


def calculate_trend_accuracy(y_true, y_pred):
    count_true = 0
    count_false = 0
    for idx in range(1, len(y_pred)):
        if (y_pred[idx] < y_true[idx - 1] and y_true[idx] < y_true[idx - 1]) or \
           (y_pred[idx] > y_true[idx - 1] and y_true[idx] > y_true[idx - 1]):
            count_true += 1
        else:
            count_false += 1
    return count_true, count_false


count_true, count_false = calculate_trend_accuracy(y_test_inv, y_pred_inv)
print(f"Trend_1: {count_false}")
print(f"Trend_2: {count_true}")

accuracy = np.mean([1 - abs(y_test_inv[i] - y_pred_inv[i]) / y_test_inv[i] for i in range(len(y_pred_inv))]) * 100
print(f"Accuracy: {accuracy}")

plt.figure(figsize=(14, 7))
plt.plot(data.index[-len(y_test):], y_test_inv, color='blue', label='Actual Values')
plt.plot(data.index[-len(y_test):], y_pred_inv, color='red', label='Predicted Values')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Actual vs Predicted Closing Prices ANN2')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
