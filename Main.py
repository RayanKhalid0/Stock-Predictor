import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

import yfinance
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(actual, predicted):
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    mae = mean_absolute_error(actual, predicted)

    def smape(actual, predicted):
        return 100 / len(actual) * np.sum(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))

    smape_val = smape(actual, predicted)

    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    return mape, mae, smape_val, mse, r2


company = 'AMZN'

start = dt.datetime(2015, 1, 1)
end = dt.datetime(2023, 1, 1)

data = yfinance.download(company, start, end)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(x_train, y_train, epochs=75, batch_size=32, validation_split=0.1, verbose=1, shuffle=False)

test_start = dt.datetime(2023, 1, 1)
test_end = dt.datetime.now()

test_data = yfinance.download(company, test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']))

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

mape, mae, smape_val, mse, r2 = evaluate_model(actual_prices, predicted_prices)

print("Mean Absolute Percentage Error (MAPE): {:.2f}%".format(mape))
print("Mean Absolute Error (MAE): {:.2f}".format(mae))
print("Symmetric Mean Absolute Percentage Error (SMAPE): {:.2f}%".format(smape_val))
print("Mean Squared Error (MSE): {:.2f}".format(mse))
print("R-squared (R^2): {:.2f}".format(r2))

plt.plot(actual_prices, color="black", label=f"Actual {company} price")
plt.plot(predicted_prices, color='green', label=f"Predicted {company} price")
plt.title(f"{company} share price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share price')
plt.legend()
plt.show()

real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0:]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")
