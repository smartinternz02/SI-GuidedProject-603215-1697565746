import pandas as pd
import numpy as np
import xgboost
import pickle
import os
from flask import Flask, render_template, url_for, request

app = Flask(__name__)
model = pickle.load(open(r'xg.model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('templates\index.html') #rendering the home page?

@app.route('/predict',methods=["POST","GET"]) # route to show the predictions in a web UI
def predict():
    # reading the inputs given by the user
    input_feature=[float(x) for x in request.form.values()]
    features_values=[np.array(input_feature)]
    names = [['playerID','Sex','Equipment','Age','BodyweightKg','BestSquatKg', 'BestBenchKg']]
    data = pd.DataFrame(features_values, column=names)
    prediction = model.predict(data)
    print(prediction)
    text = "Estimated Deadlift for the builder is: "
    return render_template("index.html", prediction_text = text + str(prediction))

if __name__ == '__main__':
    app.run(debug=True)