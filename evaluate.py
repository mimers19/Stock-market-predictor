import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from pandas.tseries.offsets import BDay
import datetime

# Wczytanie modelu
model = load_model('stock_prediction_model.h5')

# Funkcja do wczytania i przygotowania danych
def prepare_data(file_path, look_back=10):
    df = pd.read_csv(file_path, index_col=None, header=0)
    
    # Normalizacja danych
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])
    df['Scaled_Close'] = scaled_data

    data = df['Scaled_Close'].values.reshape(-1, 1)
    return data, scaler, df

# Funkcja do tworzenia datasetu
def create_dataset(dataset, look_back=1):
    dataX = []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
    return np.array(dataX)

# Funkcja do generowania prognozy na zadana liczbe dni
def generate_forecast(model, data, scaler, look_back=10, days=30):
    forecast = []
    input_seq = data[-look_back:]  # Weź ostatnie look_back dni jako dane wejściowe
    
    for _ in range(days):
        input_seq = input_seq.reshape((1, look_back, 1))
        next_value = model.predict(input_seq)
        forecast.append(next_value[0, 0])
        
        # Zaktualizuj dane wejściowe
        input_seq = np.append(input_seq[:, 1:, :], [[[next_value[0, 0]]]], axis=1)

    # Odwróć skalowanie
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    return forecast

# Funkcja do sprawdzenia, czy dzień jest dniem roboczym w USA
def is_business_day(date):
    us_holidays = [
        '2024-01-01', '2024-01-15', '2024-02-19', '2024-04-02', '2024-05-28',
        '2024-07-04', '2024-09-03', '2024-11-22', '2024-12-25'
    ]
    return np.is_busday(date.strftime('%Y-%m-%d')) and date.strftime('%Y-%m-%d') not in us_holidays

# Funkcja do generowania przyszłych dni roboczych
def generate_future_dates(start_date, num_days):
    future_dates = []
    current_date = start_date
    while len(future_dates) < num_days:
        current_date += BDay()
        if is_business_day(current_date):
            future_dates.append(current_date)
    return future_dates

# Wczytaj dane
data, scaler, df = prepare_data('your_stock_data.csv')

# Generuj prognozę na 30 dni
forecast_days = 30
forecast = generate_forecast(model, data, scaler, look_back=10, days=forecast_days)

# Generuj przyszłe dni robocze
start_date = df['Date'].iloc[-1]
start_date = pd.to_datetime(start_date)
future_dates = generate_future_dates(start_date, forecast_days)

# Tworzenie DataFrame z prognozami
forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast_Close': forecast.flatten()})

forecast_df.to_csv('stock_forecast.csv', index=False)
