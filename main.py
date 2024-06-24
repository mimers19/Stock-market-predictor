import yfinance as yf
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
from PIL import Image, ImageTk
import pandas as pd
from datetime import datetime
from tickers import tickers  # załóżmy, że tickers zawiera listę dostępnych tickerów

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

    # Sprawdzenie, czy ticker znajduje się na liście tickers
    if ticker not in tickers:
        messagebox.showerror("Błąd", f"Nieprawidłowy ticker '{ticker}'. Wprowadź poprawną nazwę akcji z listy.")
        return

    dane = pobierz_dane_akcji(ticker, start_date, end_date)
    if not dane.empty:
        rysuj_wykres(dane, ticker, model_end_date)
    else:
        tk.messagebox.showerror("Błąd", "Brak danych dla podanego zakresu dat.")

# Funkcja do utworzenia przycisku z zaokrąglonymi rogami
def create_rounded_rect_button(canvas, x, y, width, height, radius, text, command):
    points = [x+radius, y, x+width-radius, y,
              x+width, y, x+width, y+radius,
              x+width, y+height-radius, x+width, y+height,
              x+width-radius, y+height, x+radius, y+height,
              x, y+height, x, y+height-radius,
              x, y+radius, x, y]
    rect = canvas.create_polygon(points, smooth=True, fill="darkgrey", outline="black", width=3)
    label = canvas.create_text(x + width / 2, y + height / 2, text=text, font=("Century Gothic", 13, 'bold'))
    canvas.tag_bind(rect, '<ButtonPress-1>', lambda e: command())
    canvas.tag_bind(label, '<ButtonPress-1>', lambda e: command())
    return rect, label

def update_combobox(event):
    typed = ticker_entry.get().lower()
    if typed == '':
        data = tickers
    else:
        data = [item for item in tickers if typed in item.lower()]

    # Save current selection
    current_selection = ticker_entry.get()

    ticker_entry['values'] = data
    ticker_entry.set(current_selection)

    # Open the dropdown if there are matches
    if data:
        ticker_entry.event_generate('<Down>')

# Funkcja do formatowania daty do postaci "5 marca 2023"
def format_date_for_display(date):
    return date.strftime('%-d %B %Y')  # %-d usuwa wiodące zera

# Tworzenie głównego okna
root = tk.Tk()
root.title("Pobieranie danych akcji")
root.geometry("800x600")
root.resizable(False, False)  # Zablokowanie możliwości zmiany rozmiaru okna

# Ustawienie ikony aplikacji
root.iconbitmap('dollaricon.png')  # Zmień 'dollaricon.png' na nazwę Twojego pliku ikony

# Wczytanie obrazu tła
background_image = Image.open("dolary.png")
background_photo = ImageTk.PhotoImage(background_image)

# Utworzenie etykiety z obrazem tła
background_label = tk.Label(root, image=background_photo)
background_label.place(relwidth=1, relheight=1)

# Ramka dla elementów interfejsu użytkownika
frame = tk.Frame(root, bg='white', bd=5)
frame.place(relx=0.5, rely=0.1, relwidth=0.3, relheight=0.7, anchor='n')

# Definicja czcionki
font = ('Century Gothic', 12, 'bold')

# Etykieta i pole do wprowadzania tickera
ttk.Label(frame, text="Ticker akcji:", font=font).pack(pady=5)
ticker_entry = ttk.Combobox(frame, values=tickers, font=font)
ticker_entry.pack(pady=5)
ticker_entry.bind('<KeyRelease>', update_combobox)

# Etykiety i pola do wyboru zakresu dat
ttk.Label(frame, text="Data początkowa:", font=font).pack(pady=5)
start_cal = DateEntry(frame, width=12, background='darkblue',
                      foreground='white', borderwidth=2, year=2023, month=1, day=1, font=font)
start_cal.pack(pady=5)

ttk.Label(frame, text="Data końcowa:", font=font).pack(pady=5)
end_cal = DateEntry(frame, width=12, background='darkblue',
                    foreground='white', borderwidth=2, year=2023, month=12, day=31, font=font)
end_cal.pack(pady=5)

ttk.Label(frame, text="Data końcowa dla modelu:", font=font).pack(pady=5)
model_end_cal = DateEntry(frame, width=12, background='darkblue',
                          foreground='white', borderwidth=2, year=2024, month=6, day=30, font=font)
model_end_cal.pack(pady=5)

# Utworzenie kanwy dla przycisku
canvas = tk.Canvas(frame, width=200, height=50, bg='white', highlightthickness=0)
canvas.pack(pady=30)

# Dodanie przycisku z zaokrąglonymi rogami do kanwy
create_rounded_rect_button(canvas, 10, 10, 180, 30, 20, "Generuj wykres", generuj_wykres)

# Uruchomienie aplikacji
root.mainloop()
