from flask import Flask, render_template, request
# from data_preprocessing import DataPreprocessing as dp
from tensorflow.keras.models import load_model
from src.modules import data_preprocessing as dp
from sklearn.preprocessing import MinMaxScaler

import pickle 
import pandas as pd
import numpy as np

app = Flask(__name__)
#model_amd = pickle.load(open('model_amd.pkl','rb')) #read mode
#model = pickle.load(open('model.pkl','rb')) #read mode
# amd_model = load_model('model_amd.h5')
# aapl_model = load_model('model_aapl.h5')
# msft_model = load_model('model_msft.h5')
# goog_model = load_model('model_goog.h5')
# meta_model = load_model('model_meta.h5')

amd_model = load_model('models/amd_model.h5')
aapl_model = load_model('models/aapl_model.h5')
msft_model = load_model('models/msft_model.h5')
goog_model = load_model('models/goog_model.h5')
meta_model = load_model('models/meta_model.h5')

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        #access the data from form
        ## Ticker
        ticker = request.form["ticker"]

        X = []
        y = []

        #get prediction
        X, y, scaler = dp.preprocess_data(ticker)

        # if ticker == "amd":
        #     prediction = predict(X, y, model_amd, scaler)
        # elif ticker == "aapl":
        #     prediction = predict(X, y, model_aapl, scaler)
        # elif ticker == "goog":
        #     prediction = predict(X, y, model_goog, scaler)
        # elif ticker == "msft":
        #     prediction = predict(X, y, model_msft, scaler)
        # elif ticker == "meta":
        #     prediction = predict(X, y, model_meta, scaler)
        # else:
        #     prediction = 0

        if ticker == "amd":
            prediction = predict(X, y, amd_model, scaler)
        elif ticker == "aapl":
            prediction = predict(X, y, aapl_model, scaler)
        elif ticker == "goog":
            prediction = predict(X, y, goog_model, scaler)
        elif ticker == "msft":
            prediction = predict(X, y, msft_model, scaler)
        elif ticker == "meta":
            prediction = predict(X, y, meta_model, scaler)
        else:
            prediction = 0
        
        return render_template("index.html", prediction_text='Your predicted stock prices in next 5 days are $ {}'.format(prediction))

# Function to generate the future forecasts
def predict( X, y, model, scaler):

    # prepare data for forecasting
    # forecast_data = df[['adj close']].values

    # z = scaler.transform(forecast_data)

    # X, y = [], []

    # for i in range(window_size, len(z)):
    #     X.append(z[i - window_size: i])
    #     y.append(z[i])

    # X, y = np.array(X), np.array(y)
    n_past = 0
    n_future = 5
    X_past = X[- n_past - 1:, :, :][:1]  # last observed input sequence
    y_past = y[- n_past - 1]             # last observed target value
    y_future = []                        # predicted target values

    for i in range(n_past + n_future):

        # feed the last forecast back to the model as an input
        X_past = np.append(X_past[:, 1:, :], y_past.reshape(1, 1, 1), axis=1)

        # generate the next forecast
        y_past = model.predict(X_past)

        # save the forecast
        y_future.append(y_past.flatten()[0])
  
    # transform the forecasts back to the original scale
    y_future = scaler.inverse_transform(np.array(y_future).reshape(-1, 1)).flatten()

    return y_future

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)