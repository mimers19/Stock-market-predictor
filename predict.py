import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox

def predict_past(ticker):
    path = f"stock_data/{ticker}_stock_data.csv"
    print(ticker)
    df = pd.read_csv(path, index_col=None, header=0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    look_back = 100
    x_data = []
    y_data = []
    dates = []
    for i in range(look_back, len(stock_data)):
        if i > 0.95 * len(stock_data):
            x_data.append(stock_data[i - look_back:i])
            y_data.append(stock_data[i])
            dates.append(df['Date'].values[i])
    model = load_model('updated_model.keras')

    predictions = []
    for i in range(len(y_data)):
        predicted_value = model.predict(x_data[i].reshape(1, x_data[i].shape[0], x_data[i].shape[1]))
        predictions.append(predicted_value[0, 0])

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    y_data = scaler.inverse_transform(y_data)

    plt.figure(figsize=(14, 5))
    plt.plot(dates, y_data, color='blue', label='Actual Stock Price')
    plt.plot(dates, predictions, color='red', label='Predicted Stock Price')

    # Ustawianie osi X, aby wyświetlała co 10 dzień
    plt.xticks(np.arange(0, len(dates), step=10), rotation=45)

    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.tight_layout()  # Dopasowanie układu, aby daty się nie nakładały
    plt.show()  

def predict_one(ticker):
    path = f"stock_data/{ticker}_stock_data.csv"
    df = pd.read_csv(path, index_col=None, header=0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    look_back = 100
    x_data = stock_data[-look_back:].reshape(1, look_back, 1)

    model = load_model('updated_model.keras')

    predicted_value = model.predict(x_data)
    predicted_value = scaler.inverse_transform(predicted_value.reshape(-1, 1))

    last_days_prices = scaler.inverse_transform(stock_data[-look_back:]).flatten()

    # Tworzenie wiadomości z przewidywaną wartością i ostatnimi cenami
    message = f"Przewidywana wartość na jutro: {predicted_value[0, 0]:.2f} USD\n\n"
    message += "Ostatnie ceny:\n"
    for i in range(1, 6):
        message += f"{i}d: {last_days_prices[-i]:.2f} USD\n"

    # Wyświetlenie wiadomości w oknie messagebox
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Przewidywana wartość", message)

