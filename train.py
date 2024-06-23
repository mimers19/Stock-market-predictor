from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import boto3

# Inicjalizacja sesji Spark
spark = SparkSession.builder.appName('StockPrediction').getOrCreate()

# Nazwa bucketu S3
bucket_name = 'my-stock-data-pg-69-2137'

# Używanie boto3 do uzyskania listy plików w S3
s3 = boto3.client('s3')
response = s3.list_objects_v2(Bucket=bucket_name)

# Pobranie wszystkich plików CSV z bucketu
all_files = [f"s3://{bucket_name}/{item['Key']}" for item in response.get('Contents', []) if item['Key'].endswith('.csv')]

# Funkcja do ładowania i przetwarzania danych z plików CSV
def load_and_process_data(file_path):
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    window = Window.orderBy('Date')
    df = df.withColumn('Prev_Close', lag('Close', 1).over(window))
    df = df.dropna()
    return df

# Ładowanie danych ze wszystkich plików CSV
dataframes = [load_and_process_data(file) for file in all_files]
data = dataframes[0].unionAll(*dataframes[1:])

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
model_path = "output/stock_gbt_model"
model.write().overwrite().save(model_path)

spark.stop()
