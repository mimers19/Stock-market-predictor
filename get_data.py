import yfinance as yf
import pandas as pd
import os

# Rozszerzona lista tickerów firm
tickers = [
    'AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'BABA', 'V', 'JPM',
    'JNJ', 'WMT', 'PG', 'MA', 'DIS', 'PYPL', 'NFLX', 'ADBE', 'CMCSA', 'XOM',
    'KO', 'NKE', 'PFE', 'PEP', 'T', 'MRK', 'INTC', 'CSCO', 'VZ', 'WFC',
    'BA', 'CVX', 'COST', 'MCD', 'IBM', 'HON', 'SBUX', 'MMM', 'MDT', 'RTX',
    'GS', 'AMGN', 'TXN', 'GILD', 'QCOM', 'CAT', 'SPG', 'NOW', 'LRCX', 'AVGO',
    'BLK', 'BKNG', 'ISRG', 'DE', 'ADP', 'SYK', 'PLD', 'MO', 'TMO', 'CI',
    'UNH', 'ABBV', 'ABT', 'DHR', 'LIN', 'MRNA', 'REGN', 'ATVI', 'SNPS', 'CSX',
    'NSC', 'UPS', 'FDX', 'GD', 'LMT', 'NOC', 'CL', 'C', 'SO', 'NEE',
    'D', 'DUK', 'EXC', 'AEP', 'PCG', 'SRE', 'PPL', 'PEG', 'ED', 'FE'
]

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