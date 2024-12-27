import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import yfinance as yf

data = yf.download('SPY', start='2008-07-01', end='2024-07-16')
print(data.head())

print(data.isna().sum())

data.fillna(method='ffill', inplace=True)

data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()
data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
data['RSI_14'] = 100 - (100 / (1 + data['Close'].diff().apply(lambda x: np.where(x > 0, x, 0)).rolling(window=14).mean() /
                                data['Close'].diff().apply(lambda x: np.where(x < 0, -x, 0)).rolling(window=14).mean()))
data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
data['Bollinger_Upper'] = data['Close'].rolling(window=20).mean() + (data['Close'].rolling(window=20).std() * 2)
data['Bollinger_Lower'] = data['Close'].rolling(window=20).mean() - (data['Close'].rolling(window=20).std() * 2)

data.fillna(method='bfill', inplace=True)

for lag in range(1, 6):
    data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)

data.dropna(inplace=True)

features = ['Open', 'High', 'Close','Low', 'Volume', 'SMA_50', 'SMA_200', 'EMA_20', 'RSI_14', 'MACD', 'Bollinger_Upper', 'Bollinger_Lower'] + \
           [f'Close_Lag_{lag}' for lag in range(1, 6)]
data['Target'] = data['Close'].shift(-1)

print()

data.dropna(inplace=True)

X = data[features]
y = data['Target']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

X_train = pd.DataFrame(X_train, columns=features)
X_test = pd.DataFrame(X_test, columns=features)

model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

early_stop = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop])

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error on the test set: {mse}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")
print(f"R-squared (R²) on the test set: {r2}")
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Square Error: {rmse}")

count_true = 0
count_false = 0

for idx in range(1, len(y_pred)):
    if (y_pred[idx] < y_test[idx-1] and y_test.iloc[idx] < y_test.iloc[idx-1]) or (y_pred[idx] > y_test[idx-1] and y_test.iloc[idx] > y_test.iloc[idx-1]):
        count_true += 1
    else:
        count_false += 1

print(count_false)
print(count_true)

accuracy = np.mean([1 - abs(y_test.iloc[i] - y_pred[i]) / y_test.iloc[i] for i in range(len(y_pred))]) * 100
print(f"Accuracy: {accuracy}")

plt.figure(figsize=(14, 7))
plt.plot(data.index[-len(y_test):], y_test, color='blue', label='Actual Values')
plt.plot(data.index[-len(y_test):], y_pred, color='red', linestyle='dashed', label='Predicted Values')
future_index = data.index[-1] + pd.Timedelta(days=1)
plt.xlim([data.index[-len(y_test)].to_pydatetime(), future_index.to_pydatetime() + pd.Timedelta(days=1)])
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Actual vs Predicted Closing Prices ANN1')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
