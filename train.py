import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pyspark.sql import SparkSession
import boto3
import io

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("StockPricePrediction") \
    .getOrCreate()

# AWS S3 Configuration
s3_bucket = 'my-stock-data-pg-69-2137'
s3_path = 'stock_data/TSLA_stock_data.csv'
model_save_path = 's3://your-s3-bucket-name/models/updated_model.keras'



# Initialize S3 client
s3_client = boto3.client('s3')
def load_data_from_s3(bucket, path):
    response = s3_client.get_object(Bucket=bucket, Key=path)
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    if status == 200:
        print(f"Successfully fetched data from {bucket}/{path}")
        return pd.read_csv(io.BytesIO(response['Body'].read()), index_col=None, header=0)
    else:
        print(f"Failed to fetch data from {bucket}/{path}")
        return None

def train_model():
    df = load_data_from_s3(s3_bucket, s3_path)
    
    if df is None:
        return None, None, None

    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    look_back = 100
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in range(look_back, len(stock_data)):
        if i < 0.8 * len(stock_data):
            x_train.append(stock_data[i - look_back:i])
            y_train.append(stock_data[i])
        else:
            x_test.append(stock_data[i - look_back:i])
            y_test.append(stock_data[i])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)

    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=32, epochs=50)

    # Save model to S3
    with open('test_model.keras', 'wb') as model_file:
        model.save(model_file)
    s3_client.upload_file('test_model.keras', s3_bucket, 'models/test_model.keras')
    print("Model trained and saved to S3 as test_model.keras")
    
    return x_test, y_test, scaler

def test_model(x_test, y_test, scaler):
    # Load model from S3
    with open('test_model.keras', 'wb') as model_file:
        s3_client.download_fileobj(s3_bucket, 'models/test_model.keras', model_file)
    
    model = load_model('test_model.keras')

    predictions = []
    for i in range(len(y_test)):
        predicted_value = model.predict(x_test[i].reshape(1, x_test[i].shape[0], x_test[i].shape[1]))
        predictions.append(predicted_value[0, 0])

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test)

    # Plotting the results
    plt.figure(figsize=(14, 5))
    plt.plot(y_test, color='blue', label='Actual Stock Price')
    plt.plot(predictions, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    x_test, y_test, scaler = train_model()
    if x_test is not None and y_test is not None:
        test_model(x_test, y_test, scaler)
