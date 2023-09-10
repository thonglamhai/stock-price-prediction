from flask import Flask, render_template, request
from data_preprocessing import DataPreprocessing as dp

import pickle 
import pandas as pd
import numpy as np

app = Flask(__name__)
model_amd = pickle.load(open('model_amd.pkl','rb')) #read mode
model = pickle.load(open('model.pkl','rb')) #read mode

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        #access the data from form
        ## Ticker
        ticker = int(request.form["ticker"])

        #get prediction
        X = model.preprocess_data(ticker)

        if ticker == "amd":
            prediction = model_amd.predict(X)
        else:
            prediction = model.predict(X)
        
        return render_template("index.html", prediction_text='Your predicted stock price is $ {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)