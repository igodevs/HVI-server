#!/usr/local/lib/python2.7
# -*- coding: utf-8 -*- 

from flask import Flask, jsonify, request, send_file
import os
from rbf import load_data, bp_train, get_predict, err_rate, save_model_result
from flask_test import load_data, load_model, get_predict, save_predict
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods = ['GET'])
def start():
    return 'Hello'

@app.route('/sendFile', methods = ['GET'])
def sendfile():
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    pptPaht = dir_path+'/prezentacia.pptx'
    print(pptPaht)
    return send_file(pptPaht)

@app.route('/trainNN', methods = ['GET'])
def start_nn():
    print("--------- 1.load data ------------")
    feature, label, n_output = load_data("data.txt")
    print("--------- 2.training ------------")
    center, delta, w = bp_train(feature, label, 20, 5000, 0.008, n_output)
    print("--------- 3.get prediction ------------")
    result = get_predict(feature, center, delta, w)
    # print("result：", (1 - err_rate(label, result)))
    print("--------- 4.save model and result ------------")
    save_model_result(center, delta, w, result)
    return jsonify({  "res": "success" })

@app.route('/testNN', methods = ['POST'])
def test_nn():
    req_data = request.get_json()
    print("--------- 1.load data ------------")
    dataTest = load_data(
        [
        req_data['x1']
        , req_data['x2']
        , req_data['x3']
        , req_data['x4']
        , req_data['x5']
        , req_data['x6']
        , req_data['x7']
        , req_data['x8']
        , req_data['x9']
        , req_data['x10']
        , req_data['x11']
        , req_data['x12']
        , req_data['x13']
        , req_data['x14']
        , req_data['x15']
        , req_data['x16']
        , req_data['x17']
        , req_data['x18']
        , req_data['x19']
        ]
    )
    print("--------- 2.load model ------------")
    center, delta, w = load_model("messidor_center.txt", "messidor_delta.txt", "messidor_weight.txt")
    print("--------- 3.get prediction ------------")
    result = get_predict(dataTest, center, delta, w)
    print('result', result)
    print("--------- 4.save result ------------")
    res = save_predict(result)
    return jsonify({  "res": res })


if(__name__) == '__main__':
    app.run(debug = True)
