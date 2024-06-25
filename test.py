import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

def train_model():
    path = "stock_data/GOOGL_stock_data.csv"

    df = pd.read_csv(path, index_col=None, header=0)


    scaler = MinMaxScaler(feature_range = (0,1))

    stock_data = scaler.fit_transform(df['Close'].values.reshape(-1,1))


    loock_back =   100
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in range(loock_back,len(stock_data)):
        if i<0.8*len(stock_data):
            x_train.append(stock_data[i-loock_back:i])
            y_train.append(stock_data[i])
        else:
            x_test.append(stock_data[i-loock_back:i])
            y_test.append(stock_data[i])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test) , np.array(y_test)
    #print(x_train.shape,y_train.shape)
    #print(x_test.shape,y_test.shape)
    '''(3896, 100, 1) (3896, 1)
    (998, 100, 1) (998, 1)'''


    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    '''ODKOMENTOWAC NA UCZENIE'''
    #model.compile(optimizer='adam', loss='mean_squared_error')

        
    #model.fit(x_train, y_train, batch_size=32, epochs=50)
    #model.save('test_model.keras')
    print("Model trained and saved as test_model.keras")
    return x_test,y_test,scaler

def test_model(x_test,y_test,scaler):

    model = load_model('test_model.keras')
    predictions = model.predict(x_test)
    print(x_test.shape)
    # Rescaling the predictions and y_test back to original scale
    predictions = scaler.inverse_transform(predictions)
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
    x_test,y_test,scaler= train_model()
    test_model(x_test,y_test,scaler)