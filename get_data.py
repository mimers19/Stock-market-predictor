import yfinance as yf
import pandas as pd
import os
import boto3
from tickers import tickers  # Ensure you have a tickers.py file with a list named tickers
import io

# AWS S3 Configuration
s3_bucket = 'my-stock-data-pg-69-2137'


# Time period for which we retrieve data (e.g., from 2000-01-01 to today)
start_date = '2000-01-01'
end_date = '2025-05-30'

# Initialize S3 client
s3_client = boto3.client('s3')

# Download data and save to S3
for ticker in tickers:
    # Download historical data for the selected company
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Save the data to a CSV file in memory
    csv_buffer = io.StringIO()
    data.to_csv(csv_buffer)
    csv_buffer.seek(0)
    
    # Define the S3 object key (file path in the bucket)
    s3_key = f"stock_data/{ticker}_stock_data.csv"
    
    # Upload the CSV file to S3
    s3_client.put_object(Bucket=s3_bucket, Key=s3_key, Body=csv_buffer.getvalue())
    print(f"Data for {ticker} saved to S3 bucket {s3_bucket} with key {s3_key}")
