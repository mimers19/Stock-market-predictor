import yfinance as yf
import pandas as pd
import os
from tickers import tickers
# Rozszerzona lista tickerów firm


# Okres, dla którego pobieramy dane (np. od 2000-01-01 do dziś)
start_date = '2000-01-01'
end_date = '2025-05-30'

# Nazwa folderu do przechowywania plików CSV
folder_name = 'stock_data'

# Tworzenie folderu, jeśli nie istnieje
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Pobranie danych i zapisanie do plików CSV
for ticker in tickers:
    # Pobranie danych historycznych dla wybranej firmy
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Zapisanie danych do pliku CSV w stworzonym folderze
    filename = os.path.join(folder_name, f"{ticker}_stock_data.csv")
    data.to_csv(filename)
    print(f"Dane dla {ticker} zapisane do pliku {filename}")