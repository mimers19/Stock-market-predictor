from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from functools import reduce
import os

# Inicjalizacja sesji Spark
spark = SparkSession.builder.appName('StockPrediction').getOrCreate()

# Ścieżka do folderu z plikami CSV na lokalnym komputerze
folder_path = 'stock_data'

# Pobranie wszystkich plików CSV z folderu
all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

# Funkcja do ładowania i przetwarzania danych z plików CSV
def load_and_process_data(file_path):
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    window = Window.orderBy('Date')
    df = df.withColumn('Prev_Close', lag('Close', 1).over(window))
    df = df.dropna()
    return df

# Ładowanie danych ze wszystkich plików CSV
dataframes = [load_and_process_data(file) for file in all_files]
data = reduce(lambda df1, df2: df1.union(df2), dataframes)

# Przygotowanie danych do modelu
assembler = VectorAssembler(inputCols=['Open', 'High', 'Low', 'Close', 'Volume', 'Prev_Close'], outputCol='features')
data = assembler.transform(data)

# Podział danych na zestaw treningowy i testowy
train_data, test_data = data.randomSplit([0.7, 0.3])

# Stworzenie modelu Random Forest do regresji
rf = RandomForestRegressor(labelCol='Close', featuresCol='features')
model = rf.fit(train_data)

# Ocena modelu
predictions = model.transform(test_data)
evaluator = RegressionEvaluator(labelCol='Close', predictionCol='prediction', metricName='rmse')
rmse = evaluator.evaluate(predictions)
print(f"RMSE: {rmse}")

# Zapisanie modelu
model.save("model")

spark.stop()
