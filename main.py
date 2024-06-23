import yfinance as yf
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry


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


def generuj_wykres():
    ticker = ticker_entry.get()
    start_date = start_cal.get_date().strftime('%Y-%m-%d')
    end_date = end_cal.get_date().strftime('%Y-%m-%d')

    dane = pobierz_dane_akcji(ticker, start_date, end_date)
    if not dane.empty:
        rysuj_wykres(dane, ticker)
    else:
        tk.messagebox.showerror("Błąd", "Brak danych dla podanego zakresu dat.")


# Tworzenie głównego okna
root = tk.Tk()
root.title("Pobieranie danych akcji")
root.geometry("300x200")

# Etykieta i pole do wprowadzania tickera
ttk.Label(root, text="Ticker akcji:").pack(pady=5)
ticker_entry = ttk.Entry(root)
ticker_entry.pack(pady=5)

# Etykiety i pola do wyboru zakresu dat
ttk.Label(root, text="Data początkowa:").pack(pady=5)
start_cal = DateEntry(root, width=12, background='darkblue',
                      foreground='white', borderwidth=2, year=2023, month=1, day=1)
start_cal.pack(pady=5)

ttk.Label(root, text="Data końcowa:").pack(pady=5)
end_cal = DateEntry(root, width=12, background='darkblue',
                    foreground='white', borderwidth=2, year=2023, month=12, day=31)
end_cal.pack(pady=5)

# Przycisk do generowania wykresu
generate_button = ttk.Button(root, text="Generuj wykres", command=generuj_wykres)
generate_button.pack(pady=20)

# Uruchomienie aplikacji
root.mainloop()