import numpy as np

def cross_entropy(y, y_hat):
    return -np.sum(y * np.log(y_hat))

def cross_entropy_gradient(y, y_hat):
    return -y / y_hat

def mse(y, y_hat):
    return np.sum((y - y_hat) ** 2) / 2.0

def mse_gradient(y, y_hat):
    return y_hat - y

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_gradient(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    exp_x = np.exp(x - np.max(x)) 
    return exp_x / np.sum(exp_x, axis=0)

def softmax_gradient(x):
    s = softmax(x)
    return s * (np.eye(len(x)) - s).T

def relu(x):
    return np.maximum(x, 0)

def relu_gradient(x):
    return x > 0
