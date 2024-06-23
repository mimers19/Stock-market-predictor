from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# 1. Przygotowanie środowiska PySpark
spark = SparkSession.builder.appName("StockPredictionTraining").getOrCreate()

# 2. Wczytanie danych
data_path = "stock_data/*.csv"
df = spark.read.option("header", "true").option("inferSchema", "true").csv(data_path)

# 3. Przygotowanie danych
df = df.withColumn("Date", unix_timestamp(col("Date"), "yyyy-MM-dd").cast("timestamp"))
df = df.select("Date", "Open", "High", "Low", "Close", "Volume")

# Zestaw cech (features)
feature_cols = ["Open", "High", "Low", "Volume"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# Podział danych na zestawy treningowe i testowe
train_data, test_data = df.randomSplit([0.8, 0.2], seed=1234)

# 4. Budowa modelu
gbt = GBTRegressor(featuresCol="features", labelCol="Close", maxIter=4)
model = gbt.fit(train_data)

# Przeprowadź przewidywanie na danych testowych
predictions = model.transform(test_data)

# Ocena modelu
evaluator = RegressionEvaluator(labelCol="Close", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

print(f"RMSE: {rmse}")
print(f"R2: {r2}")

# Zapisanie modelu do pliku
model_path = "output/stock_gbt_model"
model.write().overwrite().save(model_path)

# Zakończenie
spark.stop()
