#!/usr/local/lib/python2.7
# -*- coding: utf-8 -*- 

from numpy import *
from math import sqrt


def load_data(file_name):
    f = open(file_name)
    feature_data = []
    a= []
    label = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split(",")
        lines.pop(0)
        for i in range(len(lines) - 1):
            if(i < 2):
                feature_tmp.append(float(lines[i]))
            elif(i == 2):
                feature_tmp.append(round(float(lines[i])/151, 3))
            elif(i == 3):
                feature_tmp.append(round(float(lines[i]) / 132, 3))
            elif(i == 4):
                feature_tmp.append(round(float(lines[i]) / 120, 3))
            elif (i ==5):
                feature_tmp.append(round(float(lines[i]) / 105, 3))
            elif (i ==6):
                feature_tmp.append(round(float(lines[i]) / 97, 3))
            elif (i ==7):
                feature_tmp.append(round(float(lines[i]) / 89, 3))
            elif (i ==8):
                feature_tmp.append(round(float(lines[i]) / 403, 3))
            elif (i ==9):
                feature_tmp.append(round(float(lines[i]) / 167, 3))
            elif (i == 10):
                feature_tmp.append(round(float(lines[i]) / 106, 3))
            elif (i == 11):
                feature_tmp.append(round(float(lines[i]) / 59, 3))
            elif (i ==12):
                feature_tmp.append(round(float(lines[i]) / 51, 3))
            elif (i == 13):
                feature_tmp.append(round(float(lines[i]) / 20, 3))
            elif (i == 14):
                feature_tmp.append(round(float(lines[i]) / 5, 3))
            elif (i == 15):
                feature_tmp.append(round(float(lines[i]) / 3, 3))
            elif (i ==16):
                feature_tmp.append(round(float(lines[i]) / 0.59, 3))
            elif (i ==17):
                feature_tmp.append(round(float(lines[i]) / 0.21, 3))
            else:
                feature_tmp.append(float(lines[i]))

        print(feature_tmp)

        #outputs
        label.append(int(lines[-1]))
        #data
        feature_data.append(feature_tmp)
        #print(feature_data)

    f.close()
    n_output = 1

    return mat(feature_data), mat(label).transpose(), n_output


def linear(x):
    return x


def hidden_out(feature, center, delta):
    m, n = shape(feature)
    m1, n1 = shape(center)
    hidden_out = mat(zeros((m, m1)))
    for i in range(m):
        for j in range(m1):
            #gaus
            hidden_out[i, j] = exp(-1.0 * (feature[i, :] - center[j, :]) * (feature[i, :] - center[j, :]).T / (
                        2 * delta[0, j] * delta[0, j]))
    return hidden_out


def predict_in(hidden_out, w):
    m = shape(hidden_out)[0]
    predict_in = hidden_out * w
    return predict_in


def predict_out(predict_in):
    result = linear(predict_in)
    return result


def bp_train(feature, label, n_hidden, maxCycle, alpha, n_output):
    m, n = shape(feature)
    center = mat(random.rand(n_hidden, n))
    center = center * (8.0 * sqrt(6) / sqrt(n + n_hidden)) - mat(ones((n_hidden, n))) * (
                4.0 * sqrt(6) / sqrt(n + n_hidden))
    delta = mat(random.rand(1, n_hidden))
    delta = delta * (8.0 * sqrt(6) / sqrt(n + n_hidden)) - mat(ones((1, n_hidden))) * (
                4.0 * sqrt(6) / sqrt(n + n_hidden))
    w = mat(random.rand(n_hidden, n_output))
    w = w * (8.0 * sqrt(6) / sqrt(n_hidden + n_output)) - mat(ones((n_hidden, n_output))) * (
                4.0 * sqrt(6) / sqrt(n_hidden + n_output))

    iter = 0
    while iter <= maxCycle:
        hidden_output = hidden_out(feature, center, delta)
        output_in = predict_in(hidden_output, w)
        output_out = predict_out(output_in)
        error = mat(label - output_out)
        for j in range(n_hidden):
            sum1 = 0.0
            sum2 = 0.0
            sum3 = 0.0
            for i in range(m):
                sum1 += error[i, :] * exp(
                    -1.0 * (feature[i] - center[j]) * (feature[i] - center[j]).T / (2 * delta[0, j] * delta[0, j])) * (
                                    feature[i] - center[j])
                sum2 += error[i, :] * exp(
                    -1.0 * (feature[i] - center[j]) * (feature[i] - center[j]).T / (2 * delta[0, j] * delta[0, j])) * (
                                    feature[i] - center[j]) * (feature[i] - center[j]).T
                sum3 += error[i, :] * exp(
                    -1.0 * (feature[i] - center[j]) * (feature[i] - center[j]).T / (2 * delta[0, j] * delta[0, j]))
            delta_center = (w[j, :] / (delta[0, j] * delta[0, j])) * sum1
            delta_delta = (w[j, :] / (delta[0, j] * delta[0, j] * delta[0, j])) * sum2
            delta_w = sum3
            center[j, :] = center[j, :] + alpha * delta_center
            delta[0, j] = delta[0, j] + alpha * delta_delta
            w[j, :] = w[j, :] + alpha * delta_w
        if iter % 10 == 0:
            cost = (1.0 / 2) * get_cost(get_predict(feature, center, delta, w) - label)
            print("\t-------- iter: ", iter, " ,cost: ", cost)
        if cost < 3:
            break
        iter += 1
    return center, delta, w


def get_cost(cost):
    m, n = shape(cost)

    cost_sum = 0.0
    for i in range(m):
        for j in range(n):
            cost_sum += cost[i, j] * cost[i, j]
    return cost_sum / 2


def get_predict(feature, center, delta, w):
    return predict_out(predict_in(hidden_out(feature, center, delta), w))


def save_model_result(center, delta, w, result):
    def write_file(file_name, source):
        f = open(file_name, "w")
        m, n = shape(source)
        for i in range(m):
            tmp = []
            for j in range(n):
                tmp.append(str(source[i, j]))
            f.write("\t".join(tmp) + "\n")
        f.close()

    write_file("messidor_center.txt", center)
    write_file("messidor_delta.txt", delta)
    write_file("messidor_weight.txt", w)
    write_file('messidor_train_result.txt', result)


def err_rate(label, pre):
    m = shape(label)[0]
    for j in range(m):
        if pre[j, 0] > 0.5:
            pre[j, 0] = 1.0
        else:
            pre[j, 0] = 0.0

    err = 0.0
    for i in range(m):
        if float(label[i, 0]) != float(pre[i, 0]):
            err += 1
    rate = err / m
    return rate


if __name__ == "__main__":
    print("--------- 1.load data ------------")
    feature, label, n_output = load_data("data.txt")
    print("--------- 2.training ------------")
    center, delta, w = bp_train(feature, label, 20, 5000, 0.008, n_output)
    print("--------- 3.get prediction ------------")
    result = get_predict(feature, center, delta, w)
    print("result：", (1 - err_rate(label, result)))
    print("--------- 4.save model and result ------------")
    save_model_result(center, delta, w, result)
