import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean
from pyspark.sql.window import Window
from pyspark.sql import functions as F

# Initialize Spark session
spark = SparkSession.builder \
    .appName("StockPricePrediction") \
    .getOrCreate()

# Read CSV files from S3 bucket
s3_bucket_path = "s3://my-stock-data-pg-69-2137/*.csv"
df = spark.read.option("header", "true").csv(s3_bucket_path)

# Extract company code from filename
df = df.withColumn('Company', F.regexp_extract(F.input_file_name(), r'([^/]+)_', 1))

# Convert necessary columns to the correct data types
df = df.withColumn('Close', col('Close').cast('double'))

# Handle missing values
df = df.fillna(method='ffill').fillna(method='bfill')

# Normalize the data
scaler = MinMaxScaler()
close_values = df.select('Close').collect()
close_values = np.array([row['Close'] for row in close_values]).reshape(-1, 1)
scaled_close = scaler.fit_transform(close_values)
df = df.withColumn('Scaled_Close', F.array(*[F.lit(float(x)) for x in scaled_close]))

# Calculate moving average
windowSpec = Window.orderBy("Date").rowsBetween(-9, 0)
df = df.withColumn('MA_10', mean(col('Close')).over(windowSpec))

# Convert Spark DataFrame to Pandas DataFrame for compatibility with TensorFlow
pandas_df = df.toPandas().dropna()

# Split the data
train_size = int(len(pandas_df) * 0.8)
train, test = pandas_df[:train_size], pandas_df[train_size:]

# Prepare the data for LSTM
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 10
trainX, trainY = create_dataset(train['Scaled_Close'].values.reshape(-1, 1), look_back)
testX, testY = create_dataset(test['Scaled_Close'].values.reshape(-1, 1), look_back)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], look_back, 1))
testX = np.reshape(testX, (testX.shape[0], look_back, 1))

print(f"trainX shape: {trainX.shape}")
print(f"trainY shape: {trainY.shape}")
print(f"testX shape: {testX.shape}")
print(f"testY shape: {testY.shape}")

# Create and fit the LSTM network
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)

# Evaluate the model
train_predict = model.predict(trainX)
test_predict = model.predict(testX)

# Reshape predictions to 2D for inverse transform
train_predict = train_predict.reshape(-1, 1)
test_predict = test_predict.reshape(-1, 1)

# Invert predictions
train_predict = scaler.inverse_transform(train_predict)
trainY = scaler.inverse_transform(trainY.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
testY = scaler.inverse_transform(testY.reshape(-1, 1))

print(f"train_predict shape: {train_predict.shape}")
print(f"trainY shape: {trainY.shape}")
print(f"test_predict shape: {test_predict.shape}")
print(f"testY shape: {testY.shape}")

# Calculate root mean squared error
train_score = np.sqrt(np.mean((train_predict - trainY[:train_predict.shape[0]]) ** 2))
test_score = np.sqrt(np.mean((test_predict - testY[:test_predict.shape[0]]) ** 2))

print(f'Train Score: {train_score:.2f} RMSE')
print(f'Test Score: {test_score:.2f} RMSE')

# Save the model
model.save('stock_prediction_model.h5')
