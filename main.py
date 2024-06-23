import yfinance as yf
import matplotlib.pyplot as plt

def pobierz_dane_akcji(ticker, start_date, end_date):
    # Pobierz dane akcji z Yahoo Finance
    dane = yf.download(ticker, start=start_date, end=end_date)
    return dane

def rysuj_wykres(dane, ticker):
    # Tworzenie wykresu
    plt.figure(figsize=(12, 6))
    plt.plot(dane['Close'], label=f'{ticker} Close Price')
    plt.title(f'Wykres cen akcji {ticker}')
    plt.xlabel('Data')
    plt.ylabel('Cena zamknięcia (Close Price)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    ticker = input("Podaj ticker akcji (np. AAPL): ")
    start_date = input("Podaj datę początkową (YYYY-MM-DD): ")
    end_date = input("Podaj datę końcową (YYYY-MM-DD): ")
    
    dane = pobierz_dane_akcji(ticker, start_date, end_date)
    if not dane.empty:
        rysuj_wykres(dane, ticker)
    else:
        print("Brak danych dla podanego zakresu dat.")
