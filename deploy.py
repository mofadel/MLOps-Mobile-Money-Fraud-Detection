#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 26/04/21 21:46
# @Author  : Mo_Fadel
# @File    : deploy.py
# @Software: PyCharm

import flask
import numpy as np
from flasgger import Swagger
import pickle as pkl

## 1- Create the app
app = flask.Flask(__name__)
swagger = Swagger(app)

## 2- Load the trained model
model = pkl.load(open('RandomForest_model_with_NumFeat.sav','rb'))
scaler = pkl.load(open('scaler.pkl', 'rb'))
print('Model Loaded Successfully !')

## 3- define our function/service
@app.route('/predict', methods=['POST'])
def predict():
    """ Endpoint taking one input
    ---
    parameters:
        - name: step
          in: query
          type: number
          required: true
        - name: amount
          in: query
          type: number
          required: true
        - name: type
          in: query
          type: string
          required: true
        - name: nameOrg
          in: query
          type: string
          required: true
        - name: nameDest
          in: query
          type: string
          required: true
        - name: oldbalanceOrg
          in: query
          type: number
          required: true
        - name: newbalanceOrg
          in: query
          type: number
          required: true
        - name: oldbalanceDest
          in: query
          type: number
          required: true
        - name: newbalanceDest
          in: query
          type: number
          required: true
        - name: isFlaggedFraud
          in: query
          type: number
          required: true

    responses:
        200:
            description: "1: Fraud, 0:Non-Fraud"
    """

    step = flask.request.args.get("step")
    type = flask.request.args.get("type")
    amount = flask.request.args.get("amount")
    oldbalanceOrg = flask.request.args.get("oldbalanceOrg")
    nameOrg = flask.request.args.get("nameOrg")
    newbalanceOrg = flask.request.args.get("newbalanceOrg")
    oldbalanceDest = flask.request.args.get("oldbalanceDest")
    nameDest = flask.request.args.get("nameDest")
    newbalanceDest = flask.request.args.get("newbalanceDest")
    isFlaggedFraud = flask.request.args.get("isFlaggedFraud")

    input_features = np.array([[step, amount, oldbalanceOrg, newbalanceOrg, oldbalanceDest, newbalanceDest, isFlaggedFraud]])#.reshape(-1,1)
    input_features = scaler.transform(input_features)
    prediction = model.predict(input_features)
    confidence = model.predict_proba(input_features)

    if prediction[0] == 0:
        pred = 'Prediction: '+str(prediction[0]) + ' (Non-Fraudulent Transaction)\n'
    else:
        pred = 'Prediction: '+str(prediction[0]) + ' (Fraudulent Transaction)\n'

    conf = 'Confidence over the prediction: '+str(confidence[0][prediction[0]] *100) + ' %'


    return pred + conf 

## 4- run the app
if __name__== '__main__':
    app.run(host='127.0.0.1', port=7000)
