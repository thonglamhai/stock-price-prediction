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

def predict(string):
    clf = pickle.load('model.pkl')

    df = clean_data(df)
    df = transform(([review_text]))
    pred = clf.predict(df)
    print(pred[0])
    if pred[0]:
        prediction = "Positive"
    else:
        prediction = "Negative"
    return prediction

