# Import tensorflow 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow import keras

# Import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 
import yfinance as yf
from collections import deque


# import 
import numpy as np
import pandas as pd


def clean_data(dataframe):

    # Drop the columns that are not needed
    clean_data = dataframe.dropna()

    # Handle outliers
    columns = dataframe.columns

    for column in columns:
        handle_outliers(clean_data, column)

    # Change column names to lower case to process easier
    dataframe.columns = dataframe.columns.str.lower()

    # Change index name to lower case to process easier    
    dataframe.index.name = dataframe.index.name.lower()

    # Get the column 'adj close' as feature
    dataframe = dataframe.iloc[:, 4:5]

    return dataframe

def download_data(symbol):
    '''
    Get the data from yahoo finance.
    '''
    # Get data of the symbol within 10 recent years.
    data = yf.download(symbol, period="10y")

    # Save the data to CSV file for latter use
    # data.to_csv('data/' + symbol + '.csv')

    # Convert the data to dataframe
    data = pd.DataFrame(data)

    return data

def detect_outliers(dataframe, column):
    '''
    # Function to detect outliers using interquatile
    '''
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = dataframe[(dataframe[column] < Q1 - 1.5*IQR) | (dataframe[column] > Q3 + 1.5*IQR)]
    return outliers


def print_outliers(dataframe, columns):
    '''
    Function to print the number of outliers in each column
    '''
    # Detect and print number of outliers for each feature
    for column in columns:
        outliers = detect_outliers(dataframe, column)
        print(f'Number of outliers in {column}: {len(outliers)}')

# Function to handle outliers by setting their values into the threshold
def handle_outliers(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5*IQR
    upper_limit = Q3 + 1.5*IQR
    dataframe[column] = dataframe[column].apply(lambda x: upper_limit if x > upper_limit else lower_limit if x < lower_limit else x)

def data_scaling(training_set):
    '''
    Function to do scaling using MinMaxScaler(0,1)
    '''
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    return training_set_scaled

def convert_series_to_supervised(dataset, window_size=60):
    '''
    Creating a sliding window
    A special data structure is needed to cover 60-time stamps, based on which RNN will predict the 61st price.
    Here the number of past timestamps is set to 60 based on experimentation. 
    Thus, X_train is a nested list, which contains lists of 60 time-stamp prices.
    y_train is a list of stock prices which is the next day stock price, corresponding to each list in X_train

    '''
    X_train = []
    y_train = []
    for i in range(window_size, len(dataset)):
        X_train.append(dataset[i-window_size: i, 0])
        y_train.append(dataset[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
    
    return X_train, y_train

def scale_data(dataframe, window_size=60):
    # prepare data for forecasting
    dataset = dataframe[['adj close']]

    scaler = MinMaxScaler(feature_range = (0, 1))
    scaled_data = scaler.fit_transform(dataset)

    X, y = [], []

    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size: i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)
    return X, y, scaler

def preprocess_data_for_prediction(ticker):

    # download data
    dataframe = download_data(ticker)

    # clean data
    dataframe = clean_data(dataframe)

    # scale data
    X, y, scaler = scale_data(dataframe)

    return X, y, scaler

def split_data(dataframe):
    '''
    Split the data into training set and test set
    '''
    # Split the data into training set and test set
    training_set = dataframe.iloc[:int(np.ceil(0.8*len(dataframe))), :]
    test_set = dataframe.iloc[int(np.ceil(0.8*len(dataframe))) - 60:, :]

    return training_set, test_set

def create_dataset(dataset, window_size=60):
    '''
    Function to create dataset for LSTM model
    '''
    dataX, dataY = [], []

    for i in range(window_size, len(dataset)):
        dataX.append(dataset[i-window_size:i, 0])
        dataY.append(dataset[i, 0])

    
    dataX = np.array(dataX)
    dataY = np.array(dataY)
 
    return dataX, dataY