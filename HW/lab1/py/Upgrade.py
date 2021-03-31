import numpy as np

# gradient descent
def UpgradeGD(Wv, bv, beta, Wn, bn, beta2, epsllon, z, dz, W, b, lr, e):
    g = z.T.dot(dz)
    W -= g*lr
    g = np.sum(dz, axis=0, keepdims=True)
    b -= g*lr
    
    return W, b, Wn, bn

# momentum
def UpgradeM(Wv, bv, beta, Wn, bn, beta2, epsllon, z, dz, W, b, lr, e):
    g = z.T.dot(dz)
    Wv = beta*Wv + g*lr
    W -= Wv
    
    g = np.sum(dz, axis=0, keepdims=True)
    bv = beta*bv + g*lr
    b -= bv
    
    return W, b, Wn, bn

# Adagrad
def UpgradeAda(Wv, bv, beta, Wn, bn, beta2, epsllon, z, dz, W, b, lr, e):
    g = z.T.dot(dz)
    Wn += np.sum(np.square(g), axis=0)
    coef = 1/pow(Wn+epsllon, 1/2)
    W -= lr*coef*g
    
    g = np.sum(dz, axis=0, keepdims=True)
    bn += np.sum(np.square(g), axis=0)
    coef = 1/pow(bn+epsllon, 1/2)
    b -= lr*coef*g
    
    return W, b, Wn, bn

# Adam
def UpgradeAdam(Wv, bv, beta, Wn, bn, beta2, epsllon, z, dz, W, b, lr, e):
    e = e+1

    g = z.T.dot(dz)
    Wv = beta*Wv + (1-beta)*(g**2)
    Wn = beta2*Wn + (1-beta2)*g
    v_hat = Wv / (1-beta**e)
    n_hat = Wn / (1-beta2**e)
    coef = n_hat/(np.sqrt(v_hat)+epsllon)
    W -= lr*coef
    
    g = np.sum(dz, axis=0, keepdims=True)
    bv = beta*bv + (1-beta)*(g**2)
    bn = beta2*bn + (1-beta2)*g
    v_hat = bv / (1-beta**e)
    n_hat = bn / (1-beta2**e)
    coef = n_hat/(np.sqrt(v_hat)+epsllon)
    b -= lr*coef
    
    return W, b, Wn, bn