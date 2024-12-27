import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.callbacks import EarlyStopping
from keras.src.optimizers import Adam
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout
import matplotlib.pyplot as plt
import yfinance as yf
import keras_tuner as kt

data = yf.download('SPY', start='2008-01-01', end='2024-07-16')
data.fillna(method='ffill', inplace=True)

data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()
data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
data['RSI_14'] = 100 - (
        100 / (1 + data['Close'].diff().apply(lambda x: np.where(x > 0, x, 0)).rolling(window=14).mean() /
               data['Close'].diff().apply(lambda x: np.where(x < 0, -x, 0)).rolling(window=14).mean()))
data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
data['Bollinger_Upper'] = data['Close'].rolling(window=20).mean() + (data['Close'].rolling(window=20).std() * 2)
data['Bollinger_Lower'] = data['Close'].rolling(window=20).std() - (data['Close'].rolling(window=20).std() * 2)

data.fillna(method='bfill', inplace=True)

features = ['Open', 'High', 'Close', 'Low', 'Volume', 'SMA_50', 'SMA_200']
target = 'Close_Target'
data['Close_Target'] = data['Close'].shift(-1)
data.dropna(inplace=True)

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(data[features])

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(data[target].values.reshape(-1, 1))

X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_scaled, test_size=0.2, shuffle=False)


def model_builder(hp):
    model = Sequential()
    model.add(Conv1D(filters=hp.Int('filters_1', min_value=32, max_value=128, step=32),
                     kernel_size=hp.Int('kernel_size_1', min_value=2, max_value=5, step=1),
                     activation='relu',
                     input_shape=(X_train.shape[1], 1)))
    model.add(Conv1D(filters=hp.Int('filters_2', min_value=32, max_value=128, step=32),
                     kernel_size=hp.Int('kernel_size_2', min_value=2, max_value=5, step=1),
                     activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=hp.Int('dense_units', min_value=50, max_value=200, step=50), activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(1))

    hp_learning_rate = hp.Choice('learning_rate', values=[5e-2, 5e-3, 1e-2, 1e-3, 1e-4])

    model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='mean_squared_error')

    return model


tuner = kt.RandomSearch(
    model_builder,
    objective='val_loss',
    max_trials=20,
    executions_per_trial=2,
    directory='tuner_results_CNN',
    project_name='cnn_tuning_SPY'
)

early_stopping = EarlyStopping(monitor='val_loss', patience=600, restore_best_weights=True)
tuner.search(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test), callbacks=[early_stopping])
model.save('CNN-SPY-Hyper2.h5')

y_pred = model.predict(X_test)

y_pred_inv = scaler_y.inverse_transform(y_pred)
y_test_inv = scaler_y.inverse_transform(y_test)

mse = mean_squared_error(y_test_inv, y_pred_inv)
mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)
mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Percentage Error: {mape}")
print(f"R-squared: {r2}")
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Square Error: {rmse}")

count_true = 0
count_false = 0

for idx in range(1, len(y_pred_inv)):
    if (y_pred_inv[idx] < y_test_inv[idx - 1] and y_test_inv[idx] < y_test_inv[idx - 1]) or \
            (y_pred_inv[idx] > y_test_inv[idx - 1] and y_test_inv[idx] > y_test_inv[idx - 1]):
        count_true += 1
    else:
        count_false += 1

print(f"False Trends: {count_false}")
print(f"True Trends: {count_true}")

accuracy = np.mean([1 - abs(y_test_inv[i] - y_pred_inv[i]) / y_test_inv[i] for i in range(len(y_pred_inv))]) * 100
print(f"Accuracy: {accuracy}")

plt.figure(figsize=(14, 7))
plt.plot(data.index[-len(y_test):], y_test_inv, color='blue', label='Actual Values')
plt.plot(data.index[-len(y_test):], y_pred_inv, color='red', label='Predicted Values')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Actual vs Predicted Closing Prices CNN')
plt.legend()
plt.grid(True)
plt.show()


X_full = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
y_pred_full = model.predict(X_full)

y_pred_full_inv = scaler_y.inverse_transform(y_pred_full)
y_full_inv = scaler_y.inverse_transform(y_scaled)

mse_full = mean_squared_error(y_full_inv, y_pred_full_inv)
mape_full = mean_absolute_percentage_error(y_full_inv, y_pred_full_inv)
r2_full = r2_score(y_full_inv, y_pred_full_inv)
mae_full = mean_absolute_error(y_full_inv, y_pred_full_inv)
rmse_full = np.sqrt(mse_full)

print(f"Mean Squared Error on full data: {mse_full}")
print(f"Mean Absolute Percentage Error (MAPE) on full data: {mape_full}")
print(f"R-squared (R²) on full data: {r2_full}")
print(f"Mean Absolute Error on full data: {mae_full}")
print(f"Root Mean Square Error on full data: {rmse_full}")


count_true = 0
count_false = 0

for idx in range(1, len(y_pred_full_inv)):
    if (y_pred_full_inv[idx] < y_full_inv[idx - 1] and y_full_inv[idx] < y_full_inv[idx - 1]) or \
            (y_pred_full_inv[idx] > y_full_inv[idx - 1] and y_full_inv[idx] > y_full_inv[idx - 1]):
        count_true += 1
    else:
        count_false += 1

print(f"False Trends: {count_false}")
print(f"True Trends: {count_true}")

accuracy = np.mean([1 - abs(y_full_inv[i] - y_pred_full_inv[i]) / y_full_inv[i] for i in range(len(y_pred_full_inv))]) * 100
print(f"Accuracy: {accuracy}")

plt.figure(figsize=(14, 7))
plt.plot(data.index, y_full_inv, color='blue', label='Actual Values')
plt.plot(data.index, y_pred_full_inv, color='red', label='Predicted Values')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Actual vs Predicted Closing Prices CNN (Full Data)')
plt.legend()
plt.grid(True)
plt.show()
