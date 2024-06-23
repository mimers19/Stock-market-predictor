import yfinance as yf
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry
from PIL import Image, ImageTk
import pandas as pd


def pobierz_dane_akcji(ticker, start_date, end_date):
    # Pobierz dane akcji z Yahoo Finance
    dane = yf.download(ticker, start=start_date, end=end_date)
    return dane


def rysuj_wykres(dane, ticker, model_end_date):
    # Tworzenie wykresu z danymi historycznymi
    plt.figure(figsize=(12, 6))
    plt.plot(dane['Close'], label=f'{ticker} Close Price')

    # Dodanie prostej kreski symulującej dane modelu
    last_date = dane.index[-1]
    model_dates = pd.date_range(start=last_date, end=model_end_date)
    model_values = [dane['Close'].iloc[-1]] * len(model_dates)
    plt.plot(model_dates, model_values, label='Model Simulation', linestyle='--')

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
    model_end_date = model_end_cal.get_date().strftime('%Y-%m-%d')

    dane = pobierz_dane_akcji(ticker, start_date, end_date)
    if not dane.empty:
        rysuj_wykres(dane, ticker, model_end_date)
    else:
        tk.messagebox.showerror("Błąd", "Brak danych dla podanego zakresu dat.")


# Tworzenie głównego okna
root = tk.Tk()
root.title("Pobieranie danych akcji")
root.geometry("800x600")
root.resizable(False, False)  # Zablokowanie możliwości zmiany rozmiaru okna

# Wczytanie obrazu tła
background_image = Image.open("dolary.png")
background_photo = ImageTk.PhotoImage(background_image)

# Utworzenie etykiety z obrazem tła
background_label = tk.Label(root, image=background_photo)
background_label.place(relwidth=1, relheight=1)

# Ramka dla elementów interfejsu użytkownika
frame = tk.Frame(root, bg='white', bd=5)
frame.place(relx=0.5, rely=0.1, relwidth=0.3, relheight=0.7, anchor='n')

# Etykieta i pole do wprowadzania tickera
ttk.Label(frame, text="Ticker akcji:").pack(pady=5)
ticker_entry = ttk.Entry(frame)
ticker_entry.pack(pady=5)

# Etykiety i pola do wyboru zakresu dat
ttk.Label(frame, text="Data początkowa:").pack(pady=5)
start_cal = DateEntry(frame, width=12, background='darkblue',
                      foreground='white', borderwidth=2, year=2023, month=1, day=1)
start_cal.pack(pady=5)

ttk.Label(frame, text="Data końcowa:").pack(pady=5)
end_cal = DateEntry(frame, width=12, background='darkblue',
                    foreground='white', borderwidth=2, year=2023, month=12, day=31)
end_cal.pack(pady=5)

ttk.Label(frame, text="Data końcowa dla modelu:").pack(pady=5)
model_end_cal = DateEntry(frame, width=12, background='darkblue',
                          foreground='white', borderwidth=2, year=2024, month=6, day=30)
model_end_cal.pack(pady=5)

# Przycisk do generowania wykresu
generate_button = ttk.Button(frame, text="Generuj wykres", command=generuj_wykres)
generate_button.pack(pady=20)

# Uruchomienie aplikacji
root.mainloop()
