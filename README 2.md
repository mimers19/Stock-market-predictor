
# Stock Price Prediction Project


## Project Overview

This project aims to predict future stock prices using historical stock data and a Long Short-Term Memory (LSTM) neural network. The main objective is to provide a reliable model that can forecast stock prices, helping investors make informed decisions based on predicted market trends.

The project encompasses data retrieval, preprocessing, model training, and evaluation. Historical stock data is collected using the Yahoo Finance API, which is then processed and used to train an LSTM model. The trained model is capable of predicting future stock prices based on the historical patterns it has learned.

### AWS Integration

To leverage scalable and high-performance computing resources, the model training process was carried out using AWS (Amazon Web Services). AWS provides a suite of cloud services that are ideal for running machine learning workloads, ensuring efficient and fast model training.

Using AWS, the project benefits from:
- Scalable computing power that adjusts based on the training workload.
- Robust storage solutions for handling large datasets and model files.
- Enhanced security features ensuring the safe handling of data and model artifacts.
## Technologies Used

- Python
- TensorFlow
- Keras
- Pandas
- NumPy
- Scikit-learn
- Yahoo Finance API
- AWS (Amazon Web Services)

## Project Structure

- `evaluate.py`
- `get_data.py`
- `main.py`
- `tickers.py`
- `train.py`
- `stock_prediction_model.h5`
- `your_stock_data.csv`

### evaluate.py

This script is responsible for loading the pre-trained LSTM model and using it to generate future stock price predictions. It includes functions to prepare the data, create datasets, generate forecasts, and check for business days.

**Key Functions:**
- `prepare_data(file_path, look_back=10)`: Prepares the stock data for prediction.
- `create_dataset(dataset, look_back=1)`: Creates a dataset suitable for LSTM input.
- `generate_forecast(model, data, scaler, look_back=10, days=30)`: Generates stock price forecasts for a specified number of days.
- `is_business_day(date)`: Checks if a given date is a business day in the US.
- `generate_future_business_days(start_date, days)`: Generates a list of future business days.
- `evaluate(file_path, days=30)`: Evaluates and saves the forecasted stock prices.

### get_data.py

This script retrieves historical stock data for specified tickers using the Yahoo Finance API and saves it as a CSV file.

**Key Functions:**
- `get_data(tickers, start_date, end_date, output_file)`: Fetches historical stock data for given tickers within the specified date range and saves it to a CSV file.

### main.py

This is the main script to be executed to perform the entire workflow from data retrieval to model training and evaluation. It coordinates between other scripts to ensure the process is seamless.

**Key Functions:**
- `main()`: The entry point of the project which orchestrates data retrieval, training, and evaluation.

### tickers.py

This script defines the list of stock tickers that are used in the project.

**Key Variables:**
- `tickers`: A list of stock tickers to be used for data retrieval and prediction.

### train.py

This script is responsible for training the LSTM model using the historical stock data. It includes functions to create the LSTM model, train it, and evaluate its performance.

**Key Functions:**
- `prepare_data(file_path, look_back=10)`: Prepares the stock data for training.
- `create_dataset(dataset, look_back=1)`: Creates a dataset suitable for LSTM input.
- `train_model(trainX, trainY, look_back)`: Trains the LSTM model using the prepared data.
- `evaluate_model(model, trainX, trainY, testX, testY, scaler)`: Evaluates the model's performance and calculates the RMSE for training and test datasets.
- `main()`: The main function to execute the training and evaluation process.

### stock_prediction_model.h5

This is the pre-trained LSTM model saved in HDF5 format. It is used by `evaluate.py` to generate stock price forecasts.

### your_stock_data.csv

This CSV file contains historical stock data used for training and evaluation. It includes columns like `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, etc.

## Usage

1. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Retrieve historical data**:
    ```bash
    python get_data.py
    ```

3. **Train the model**:
    ```bash
    python train.py
    ```

4. **Evaluate the model and generate forecasts**:
    ```bash
    python evaluate.py
    ```

5. **Run the main script**:
    ```bash
    python main.py
    ```

## Acknowledgments

- The Yahoo Finance API for providing historical stock data.
- TensorFlow and Keras for the machine learning framework.

## Authors 
- Michał Jankowski
- Dominik Gołembowski
- Mateusz Debis
- Maksymlian Anzulewicz
- Michał Wera
- Bartosz Hermanowski
- Adam Hinc
- Patryk Kosmalski
- Mateusz Gosciniecki
- Jakub Ławicki
