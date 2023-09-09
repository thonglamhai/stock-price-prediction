
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

def def_xgboodst_model():
    model = xgb.XGBRegressor(n_estimators=1000, max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.9, random_state=42)
    return model

def def_randomforest_model():
    model = RandomForestRegressor(n_estimators=1000, max_depth=8, random_state=42)
    return model

def def_linear_model():
    model = LinearRegression()
    return model

def def_svr_model():
    model = SVR(kernel='rbf', C=1e3, gamma=0.1)
    return model

def def_knn_model():
    model = KNeighborsRegressor(n_neighbors=2)
    return model

def def_lasso_model():
    model = Lasso(alpha=0.1)
    return model
def def_ridge_model():
    model = Ridge(alpha=0.1)
    return model
