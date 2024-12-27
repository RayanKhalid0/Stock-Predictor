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

data = yf.download('MSFT', start='2020-01-01', end='2024-06-27')

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

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'EMA_20', 'RSI_14'] + \
           [f'Close_Lag_{lag}' for lag in range(1, 6)]
target = 'Close'

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


history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on the test set: {mse}")


mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Percentage Error (MAPE): {mape}")
print(f"R-squared (R²) on the test set: {r2}")

sequential_pred = []
sequential_actual = []

for idx, row in X_test.iterrows():
    row_array = row.values.reshape(1, -1)
    pred = model.predict(row_array)
    sequential_pred.append(pred[0][0])

    sequential_actual.append(y_test.values[idx])

    model.fit(row_array, np.array([[y_test.values[idx]]]), epochs=1, verbose=0)

count_true = 0
count_false = 0

for idx, val in enumerate(sequential_actual):
    if idx!=0:
        if val < sequential_actual[idx-1] and sequential_pred[idx] < sequential_pred[idx-1]:
            count_true += 1
        elif val > sequential_actual[idx-1] and sequential_pred[idx] > sequential_pred[idx-1]:
            count_true += 1
        else:
            count_false += 1

print(count_false)
print(count_true)



plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test.values, label='Actual Prices')
plt.plot(y_test.index, sequential_pred, label='Sequential Predictions')
plt.title('Sequential Predictions vs Actual Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


