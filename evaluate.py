from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressionModel

# 1. Przygotowanie środowiska PySpark
spark = SparkSession.builder.appName("StockPredictionInference").getOrCreate()

# 2. Wczytanie modelu
model_path = "output/stock_gbt_model"
model = GBTRegressionModel.load(model_path)

# 3. Wczytanie nowych danych
# Przyklad: wczytanie nowego pliku CSV z danymi
new_data_path = "new_stock_data.csv"
new_df = spark.read.option("header", "true").option("inferSchema", "true").csv(new_data_path)

# Przygotowanie nowych danych
new_df = new_df.withColumn("Date", unix_timestamp(col("Date"), "yyyy-MM-dd").cast("timestamp"))
new_df = new_df.select("Date", "Open", "High", "Low", "Close", "Volume")

# Zestaw cech (features)
feature_cols = ["Open", "High", "Low", "Volume"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
new_df = assembler.transform(new_df)

# Przewidywanie cen akcji
predictions = model.transform(new_df)

# Wyświetlenie wyników
predictions.select("Date", "prediction").show()

# Zapisanie wyników do pliku CSV
predictions.select("Date", "prediction").write.csv("output/predicted_stock_prices.csv", header=True)

# Zakończenie
spark.stop()
