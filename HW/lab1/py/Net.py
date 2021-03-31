import numpy as np

def forward(x, W, b):
    return np.dot(x, W) + b

def GetActF(x, W, b, l, Activation):
    a = forward(x, W, b)
    z = Activation(a, l)
    return z

def HL(InputLayerSize, OuputLayerSize):
    W = np.random.uniform(-1, 1, size=(InputLayerSize, OuputLayerSize))
    b = np.random.uniform(-1, 1, size=(1, OuputLayerSize))
    Wv = 0
    bv = 0
    Wn = 0
    bn = 0
    return W, b, Wv, bv, Wn, bn