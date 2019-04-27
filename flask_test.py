#!/usr/local/lib/python2.7
# -*- coding: utf-8 -*- 

from numpy import *
from rbf import get_predict


def load_data(arr):
    feature_data = []
    count = 0
    feature_tmp = []
    for i in arr:
        if (count < 2):
            feature_tmp.append(float(i))
        elif (count == 2):
            feature_tmp.append(round(float(i) / 151, 3))
        elif (count == 3):
            feature_tmp.append(round(float(i) / 132, 3))
        elif (count == 4):
            feature_tmp.append(round(float(i) / 120, 3))
        elif (count == 5):
            feature_tmp.append(round(float(i) / 105, 3))
        elif (count == 6):
            feature_tmp.append(round(float(i) / 97, 3))
        elif (count == 7):
            feature_tmp.append(round(float(i) / 89, 3))
        elif (count == 8):
            feature_tmp.append(round(float(i) / 403, 3))
        elif (count == 9):
            feature_tmp.append(round(float(i) / 167, 3))
        elif (count == 10):
            feature_tmp.append(round(float(i) / 106, 3))
        elif (count == 11):
            feature_tmp.append(round(float(i) / 59, 3))
        elif (count == 12):
            feature_tmp.append(round(float(i) / 51, 3))
        elif (count == 13):
            feature_tmp.append(round(float(i) / 20, 3))
        elif (count == 14):
            feature_tmp.append(round(float(i) / 5, 3))
        elif (count == 15):
            feature_tmp.append(round(float(i) / 3, 3))
        elif (count == 16):
            feature_tmp.append(round(float(i) / 0.59, 3))
        elif (count == 17):
            feature_tmp.append(round(float(i) / 0.21, 3))
        else:
            feature_tmp.append(float(i))
        count = count + 1
    feature_data.append(feature_tmp)
        #print('feature data', feature_data)
    return mat(feature_data)

def load_model(file_center, file_delta, file_w):
    def get_model(file_name):
        f = open(file_name)
        model = []
        for line in f.readlines():
            lines = line.strip().split("\t")
            model_tmp = []
            for x in lines:
                model_tmp.append(float(x.strip()))
            model.append(model_tmp)
        f.close()
        return mat(model)

    center = get_model(file_center)

    delta = get_model(file_delta)

    w = get_model(file_w)

    return center, delta, w

def save_predict(pre):
    m = shape(pre)[0]
    result = []
    for i in range(m):
        if(pre[i, 0] < 0.5):
            pre[i, 0] = 0
        else:
            pre[i, 0] = 1
        result.append(str(pre[i, 0]))
    print(result)
    return result

if __name__ == "__main__":
    print("--------- 1.load data ------------")
    dataTest = load_data(x, y)
    print(dataTest)
    print("--------- 2.load model ------------")
    center, delta, w = load_model("messidor_center.txt", "messidor_delta.txt", "messidor_weight.txt")
    print("--------- 3.get prediction ------------")
    result = get_predict(dataTest, center, delta, w)
    print("--------- 4.save result ------------")
    save_predict(result)