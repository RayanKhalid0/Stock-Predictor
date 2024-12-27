# Import required libraries
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.src.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from keras.src.utils import to_categorical


def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data


# Step 2: Data Preparation
def prepare_data(data):
    # Calculate the price direction (up = 1, down = -1, neutral = 0)
    data['Price_Change'] = data['Close'].diff().shift(-1)
    data['Direction'] = 0
    data.loc[data['Price_Change'] > 0, 'Direction'] = 1
    data.loc[data['Price_Change'] < 0, 'Direction'] = -1
    data.dropna(inplace=True)

    data = pd.DataFrame(data)
    data.to_csv("Trend_Data.csv")

    # Normalize the features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])

    # Create X (features) and y (labels)
    X = []
    y = []
    look_back = 60  # Use past 60 days to predict the next day's direction
    for i in range(look_back, len(scaled_data) - 1):
        X.append(scaled_data[i - look_back:i])
        y.append(data['Direction'].iloc[i])

    X = np.array(X)
    y = np.array(y)
    y = to_categorical(y + 1, num_classes=3)  # Convert -1, 0, 1 to categorical (3 classes)

    return X, y


# Step 3: Build Hybrid CNN + LSTM Model
def build_model(timesteps, features):
    model = Sequential()

    # 1. CNN Layer
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(timesteps, features)))
    model.add(MaxPooling1D(pool_size=2))

    # 2. LSTM Layer
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))

    # 3. Fully Connected Layer
    model.add(Dense(50, activation='relu'))

    # 4. Output Layer (Softmax for 3-class classification)
    model.add(Dense(3, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Step 4: Train and Evaluate the Model
def train_and_evaluate(X_train, y_train, X_val, y_val):
    model = build_model(timesteps=X_train.shape[1], features=X_train.shape[2])
    early_stopping = EarlyStopping(monitor='val_loss', patience=1000, restore_best_weights=True, verbose=0)


    history = model.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_val, y_val))

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.show()

    return model


# Step 5: Output Evaluation Metrics
def evaluate_model(model, X_test, y_test):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


# Main Execution
if __name__ == "__main__":
    # Parameters
    ticker = 'AAPL'
    start_date = '2018-01-01'
    end_date = '2023-01-01'

    # Download and prepare data
    stock_data = download_stock_data(ticker, start_date, end_date)
    X, y = prepare_data(stock_data)

    # Train-Test Split (70% train, 30% test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

    # Train the model
    model = train_and_evaluate(X_train, y_train, X_val, y_val)

    # Evaluate the model on test data
    evaluate_model(model, X_test, y_test)
