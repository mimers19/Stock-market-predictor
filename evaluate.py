import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressionModel
from datetime import timedelta

# Inicjalizacja sesji Spark
spark = SparkSession.builder.appName('StockForecasting').getOrCreate()

# Ścieżka do pliku CSV z danymi historycznymi
data_path = 'your_stock_data.csv'

# Ładowanie danych
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Przygotowanie danych
df = df.withColumn('Prev_Close', lag('Close', 1).over(Window.orderBy('Date')))
df = df.dropna()

# Inicjalizacja VectorAssembler
assembler = VectorAssembler(inputCols=['Open', 'High', 'Low', 'Close', 'Volume', 'Prev_Close'], outputCol='features')
data = assembler.transform(df)

# Wczytanie wytrenowanego modelu
model = RandomForestRegressionModel.load('model/output')

# Funkcja do generowania predykcji dla przyszłych dni
def generate_predictions(df, model, days):
    predictions = []
    
    schema = df.schema
    for _ in range(days):
        features = assembler.transform(df).select('features').tail(1)[0].features
        prediction = model.predict(features)
        last_row = df.tail(1)[0]
        next_row = last_row.asDict().copy()
        
        # Aktualizacja wartości cech na podstawie przewidywań
        next_row['Open'] = float(next_row['Close'])  # Możesz dodać bardziej zaawansowaną logikę
        next_row['High'] = float(next_row['Close']) * 1.01  # Przykładowa logika dla High
        next_row['Low'] = float(next_row['Close']) * 0.99  # Przykładowa logika dla Low
        next_row['Volume'] = int(next_row['Volume'])  # Możesz dodać bardziej zaawansowaną logikę
        next_row['Prev_Close'] = float(last_row['Close'])
        next_row['Close'] = prediction  # Predykcja dokładnej wartości
        
        # Ustawienie nowej daty jako obiekt datetime
        next_row['Date'] = last_row['Date'] + timedelta(days=1)
        
        # Tworzenie DataFrame z nowym wierszem i wymuszenie zgodności schematu
        new_row_df = spark.createDataFrame([next_row], schema=schema)
        df = df.union(new_row_df)
        predictions.append(next_row)
        
    return predictions

# Liczba dni do przewidzenia
days_to_predict = 10  # Możesz zmienić tę wartość

predictions = generate_predictions(df, model, days_to_predict)

# Konwersja predykcji do Pandas DataFrame
preds_df = pd.DataFrame(predictions)

# Zapisywanie predykcji do pliku CSV
output_csv_path = 'output.csv'
preds_df.to_csv(output_csv_path, index=False)

spark.stop()
