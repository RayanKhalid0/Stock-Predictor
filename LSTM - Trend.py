import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import BinaryCrossentropy

data = yf.download('SPY', start='2008-01-01', end='2024-07-13')

trends = []
for idx in range(1, len(data)):
    if data.iloc[idx]['Close'] > data.iloc[idx-1]['Close']:
        trends.append(1)
    else:
        trends.append(0)

data['Trend'] = [np.nan] + trends

data = data.dropna()

scaler = MinMaxScaler()
data['Close'] = scaler.fit_transform(data[['Close']])

def create_sequences(data, trend, seq_length=10):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(trend[i])
    return np.array(X), np.array(y)

seq_length = 10
X, y = create_sequences(data['Close'].values, data['Trend'].values, seq_length)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

y_pred = (model.predict(X_test) > 0.5).astype(int)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Trend')
plt.plot(y_pred, label='Predicted Trend', alpha=0.7)
plt.title('Predicted vs Actual Trend')
plt.legend()
plt.show()

correct_predictions = np.sum(y_test == y_pred.flatten())
incorrect_predictions = np.sum(y_test != y_pred.flatten())
print(incorrect_predictions, correct_predictions)

labels = ['Correct', 'Incorrect']
values = [correct_predictions, incorrect_predictions]

plt.figure(figsize=(6, 6))
plt.bar(labels, values, color=['green', 'red'])
plt.title('Right vs Wrong Predictions')
plt.show()
