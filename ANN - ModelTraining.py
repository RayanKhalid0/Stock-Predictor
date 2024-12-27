import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import yfinance as yf

data = yf.download('SPY', start='2008-07-01', end='2024-07-04')
print(data.head())
print(data.head(-1))

print(data.isna().sum())

data.fillna(method='ffill', inplace=True)

data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()
data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
data['RSI_14'] = 100 - (
        100 / (1 + data['Close'].diff().apply(lambda x: np.where(x > 0, x, 0)).rolling(window=14).mean() /
               data['Close'].diff().apply(lambda x: np.where(x < 0, -x, 0)).rolling(window=14).mean()))

data['Close_Target'] = data['Close'].shift(-1)
data.fillna(method='bfill', inplace=True)


for lag in range(1, 6):
    data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)

data.dropna(inplace=True)

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'EMA_20', 'RSI_14'] + \
           [f'Close_Lag_{lag}' for lag in range(1, 6)]
target = 'Close_Target'
X = data[features]
y = data[target]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

X_train = pd.DataFrame(X_train, columns=features)
X_test = pd.DataFrame(X_test, columns=features)

model = Sequential()

model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

early_stop = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop])
# history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on the test set: {mse}")

mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Percentage Error (MAPE): {mape}")
print(f"R-squared (R²) on the test set: {r2}")

last_row = X_test.iloc[-1].values.reshape(1, -1)
next_value_pred = model.predict(last_row)

print(f"Predicted next value: {next_value_pred[0][0]}")

count_true = 0
count_false = 0

for idx, val in enumerate(y_pred):
    if idx != 0:
        if val < y_pred[idx - 1] and y_test[idx] < y_test[idx - 1]:
            count_true += 1
        elif val > y_pred[idx - 1] and y_test[idx] > y_test[idx - 1]:
            count_true += 1
        else:
            count_false += 1

print(count_false)
print(count_true)

accuracy = 0
for idx in range(0, len(y_pred)):
    accuracy += (1 - (abs(y_test[idx] - y_pred[idx]) / y_test[idx])) * 100

print(f"Accuracy: {accuracy / len(y_pred)}")

plt.figure(figsize=(14, 7))

plt.plot(data.index[-len(y_test):], y_test, color='blue', label='Actual Values')

plt.plot(data.index[-len(y_test):], y_pred, color='red', linestyle='dashed', label='Predicted Values')

future_index = data.index[-1] + pd.Timedelta(days=1)
plt.scatter([future_index], [next_value_pred[0][0]], color='green', label='Next Predicted Value')

plt.xlim([data.index[-len(y_test)].to_pydatetime(), future_index.to_pydatetime() + pd.Timedelta(days=1)])

plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Actual vs Predicted Closing Prices')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import yfinance as yf

data = yf.download('SPY', start='2008-01-01', end='2023-07-02')
data = yf.download('SPY', start='2008-07-01', end='2024-07-04')
print(data.head())
print(data.head(-1))

print(data.isna().sum())

data.fillna(method='ffill', inplace=True)

data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()
data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
data['RSI_14'] = 100 - (
        100 / (1 + data['Close'].diff().apply(lambda x: np.where(x > 0, x, 0)).rolling(window=14).mean() /
               data['Close'].diff().apply(lambda x: np.where(x < 0, -x, 0)).rolling(window=14).mean()))

data.fillna(method='bfill', inplace=True)

for lag in range(1, 6):
    data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)

data.dropna(inplace=True)
for idx, val in data.iterrows():
    if idx != len(data) - 1:
        data[idx]['Close_next'] = data[idx + 1]['Close']
    else:
        data[-1]['Close_next'] = 0

features = ['Open', 'High', 'Close', 'Low', 'Volume', 'SMA_50', 'SMA_200', 'EMA_20', 'RSI_14'] + \
           [f'Close_Lag_{lag}' for lag in range(1, 6)]
target = 'Close_next'

X = data[features]
y = data[target]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

X_train = pd.DataFrame(X_train, columns=features)
X_test = pd.DataFrame(X_test, columns=features)

model = Sequential()

model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

early_stop = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop])
# history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on the test set: {mse}")

mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Percentage Error (MAPE): {mape}")
print(f"R-squared (R²) on the test set: {r2}")

last_row = X_test.iloc[-1].values.reshape(1, -1)
next_value_pred = model.predict(last_row)

print(f"Predicted next value: {next_value_pred[0][0]}")
print(y_pred[-1])


count_true = 0
count_false = 0

for idx in range(1, len(y_pred)):
    if (y_pred[idx] < y_test[idx - 1] and y_test[idx] < y_test[idx - 1]) or \
            (y_pred[idx] > y_test[idx - 1] and y_test[idx] > y_test[idx - 1]):
        count_true += 1
    else:
        count_false += 1

print(f"False Trends: {count_false}")
print(f"True Trends: {count_true}")

accuracy = 0
for idx in range(0, len(y_pred)):
    accuracy += (1 - (abs(y_test[idx] - y_pred[idx]) / y_test[idx])) * 100

print(f"Accuracy: {accuracy / len(y_pred)}")

plt.figure(figsize=(14, 7))

plt.plot(data.index[-len(y_test):], y_test, color='blue', label='Actual Values')

plt.plot(data.index[-len(y_test):], y_pred, color='red', linestyle='dashed', label='Predicted Values')

future_index = data.index[-1] + pd.Timedelta(days=1)
plt.scatter([future_index], [next_value_pred[0][0]], color='green', label='Next Predicted Value')

plt.xlim([data.index[-len(y_test)].to_pydatetime(), future_index.to_pydatetime() + pd.Timedelta(days=1)])

plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Actual vs Predicted Closing Prices')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
