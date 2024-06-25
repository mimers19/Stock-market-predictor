import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import boto3
import io

# AWS S3 Configuration
s3_bucket = 'your-s3-bucket-name'
existing_model_file = 'models/updated_model.keras'
new_data_file = 'stock_data/IBM_stock_data.csv'
updated_model_file = 'models/updated_model.keras'



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

def load_and_train_model(existing_model_s3_path, new_data_s3_path):
    # Load existing model from S3
    with open('existing_model.keras', 'wb') as model_file:
        s3_client.download_fileobj(s3_bucket, existing_model_s3_path, model_file)
    model = load_model('existing_model.keras')

    # Load new data from S3
    df = load_data_from_s3(s3_bucket, new_data_s3_path)
    
    if df is None:
        return

    # Normalize data using the same scaler as used during original training
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df['Close'].values.reshape(-1, 1))
    stock_data = scaler.transform(df['Close'].values.reshape(-1, 1))

    # Prepare data for training
    look_back = 100
    x_train = []
    y_train = []

    for i in range(look_back, len(stock_data)):
        x_train.append(stock_data[i - look_back:i])
        y_train.append(stock_data[i])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Compile and train the model on new data
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=32, epochs=10)

    # Save the updated model to S3
    model.save('updated_model.keras')
    s3_client.upload_file('updated_model.keras', s3_bucket, updated_model_file)
    print(f"Updated model saved to S3 as {updated_model_file}")

if __name__ == "__main__":
    load_and_train_model(existing_model_file, new_data_file)
