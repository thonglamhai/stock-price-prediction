import pickle 
# Import tensorflow 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow import keras


def def_lstm_model(units, X_train):
    model  = Sequential()
    # add 1st lstm layer
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 5)))
    model.add(Dropout(rate = 0.2))

    # add 2nd lstm layer: 50 neurons
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(rate = 0.2))

    # add 3rd lstm layer
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(rate = 0.2))

    # add 4th lstm layer
    model.add(LSTM(units = 50, return_sequences = False))
    model.add(Dropout(rate = 0.2))

    # add output layer
    model.add(Dense(units = 1))

    # add model compile
    model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.MeanAbsoluteError()])
    
    return model

def create_model(X_train, y_train):

    # Initialize the model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    return model