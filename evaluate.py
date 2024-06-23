import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressionModel


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
model = RandomForestRegressionModel.load('output')

# Generowanie predykcji dla przyszłych dni
def generate_predictions(df, model, days):
    predictions = []
    for _ in range(days):
        features = assembler.transform(df).select('features').tail(1)[0].features
        prediction = model.predict(features)
        last_row = df.tail(1)[0]
        next_row = last_row.asDict().copy()
        next_row['Date'] = pd.to_datetime(next_row['Date']) + pd.Timedelta(days=1)
        next_row['Prev_Close'] = last_row['Close']
        next_row['Close'] = prediction  # Predykcja dokładnej wartości
        df = df.union(spark.createDataFrame([next_row]))
        predictions.append(next_row)
    return predictions

# Liczba dni do przewidzenia
days_to_predict = 30  # Możesz zmienić tę wartość

predictions = generate_predictions(df, model, days_to_predict)

# Konwersja predykcji do Pandas DataFrame
preds_df = pd.DataFrame(predictions)

# Zapisywanie predykcji do pliku CSV
output_csv_path = 'output.csv'
preds_df.to_csv(output_csv_path, index=False)
