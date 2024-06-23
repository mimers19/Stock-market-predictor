import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Inicjalizacja sesji Spark
spark = SparkSession.builder.appName('StockPrediction').getOrCreate()

# Ścieżka do folderu z plikami CSV
data_folder = 's3://my-stock-data-pg-69-2137/'

# Funkcja do ładowania i przetwarzania danych z plików CSV
def load_and_process_data(file_path):
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    window = Window.orderBy('Date')
    df = df.withColumn('Prev_Close', lag('Close', 1).over(window))
    df = df.dropna()
    return df

# Ładowanie danych ze wszystkich plików CSV
all_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
dataframes = [load_and_process_data(file) for file in all_files]
data = dataframes[0].unionAll(*dataframes[1:])

# Przygotowanie danych do modelu
assembler = VectorAssembler(inputCols=['Open', 'High', 'Low', 'Close', 'Volume', 'Prev_Close'], outputCol='features')
data = assembler.transform(data)

# Podział danych na zestaw treningowy i testowy
train_data, test_data = data.randomSplit([0.7, 0.3])


rf = RandomForestRegressor(labelCol='Close', featuresCol='features')
model = rf.fit(train_data)

# Ocena modelu
predictions = model.transform(test_data)
evaluator = RegressionEvaluator(labelCol='Close', predictionCol='prediction', metricName='rmse')
rmse = evaluator.evaluate(predictions)
print(f"RMSE: {rmse}")

# Zapisanie modelu
model.save('s3://my-stock-data-pg-69-2137/stock_prediction_model')
