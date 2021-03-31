import numpy as np

# sigmoid
def sigmoid(x, l):
    return 1.0/(1.0 + np.exp(-x))

def derivative_sigmoid(x, l):
    return np.multiply(x, 1.0-x)

#leaky relu
def leaky_relu(x, l):
    return np.where(x > 0, x, x*l)

def derivative_leaky_relu(x, l):
    return np.where(x > 0, 1, l)

# tanh
def tanh(x, l):
    return np.tanh(x)

def derivative_tanh(x, l):
    return 1 - np.square(np.tanh(x))

# linear
def linear(x, l):
    return x

def derivative_linear(x, l):
    return 1