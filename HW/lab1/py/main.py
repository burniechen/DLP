import numpy as np
import math

from Other import *
from Activation import *
from Upgrade import *
from Net import *

def Net(x, y, lr, epoch=2000, HLsize=100, beta=0.9, beta2=0.1, n=100, epsllon=1e-8, Activation=sigmoid, Derivative=derivative_sigmoid, l=0, Upgrade=UpgradeGD):
    InputL, FirstL, SecondL, OutputL = x.shape[1], HLsize, HLsize, y.shape[1]

    W1, b1, Wv1, bv1, Wn1, bn1 = HL(InputL, FirstL)
    W2, b2, Wv2, bv2, Wn2, bn2 = HL(FirstL, SecondL)
    Wout, bout, Wvout, bvout, Wnout, bnout = HL(SecondL, OutputL)

    loss_m = []
    epoch_m = []

    for e in range(epoch):
        loss = 0

        # forward
        z1 = GetActF(x , W1, b1, l, Activation)
        z2 = GetActF(z1, W2, b2, l, Activation)
        pred_y = GetActF(z2, Wout, bout, l, Activation=sigmoid)

        # backpropagation
        error = pred_y - y
        d_pred_y = 2*error * derivative_sigmoid(pred_y, l)

        error_a2 = d_pred_y.dot(Wout.T)
        d_z2 = error_a2 * Derivative(z2, l)

        error_a1 = d_z2.dot(W2.T)
        d_z1 = error_a1 * Derivative(z1, l)

        # update weights & biases
        Wout, bout, Wnout, bnout = Upgrade(Wvout, bvout, beta, Wnout, bnout, beta2, epsllon, z2, d_pred_y, Wout, bout, lr, e)
        W2, b2, Wn2, bn2 = Upgrade(Wv2, bv2, beta, Wn2, bn2, beta2, epsllon, z1, d_z2, W2, b2, lr, e)
        W1, b1, Wn1, bn1 = Upgrade(Wv2, bv2, beta, Wn1, bn1, beta2, epsllon, x, d_z1, W1, b1, lr, e)

        # print epoch & loss
        for err in error:
            loss += err[0]*err[0]

        if e%100 == 0:
            epoch_m.append(e)
            loss_m.append(loss/y.shape[0])
            
            tmp_y = pred_y
            for ele in tmp_y:
                if ele[0] > 0.5:
                    ele[0] = 1
                else:
                    ele[0] = 0

            count = 0
            for i in range(y.shape[0]):
                if tmp_y[i] == y[i]:
                    count += 1
            print(f'epoch: {e}, loss: {loss}, acc: {count/y.shape[0]}')

    for ele in pred_y:
        if ele[0] > 0.5:
            ele[0] = 1
        else:
            ele[0] = 0

    count = 0
    for i in range(y.shape[0]):
        if pred_y[i] == y[i]:
            count += 1

    show_result(x, y, pred_y)
    show_loss(epoch_m, loss_m)
    
    acc = count/y.shape[0]
    
    return pred_y, acc

np.random.seed(5)
x, y = generate_XOR_easy()
# x, y = generate_linear()
pred_y, acc = Net(x, y, lr=0.1, epoch=5000, HLsize=4, Activation=sigmoid, Derivative=derivative_sigmoid, Upgrade=UpgradeGD)
print(f'Accuracy: {acc}')
