import numpy as np

"""
For all function in this file

Args:
    y:  output, shape=(N, 1)
    tx: data sample, hape=(N, D)
    w:  weight, shape=(D, 1) 
"""

def compute_error(y, tx, w):
    """Computes the error.
    Returns:
        a loss"""
    error = y - tx @ w
    return error 


def compute_mse(y, tx, w):
    """Computes the loss using MSE.
    Returns:
        a loss"""
    #N = y.shape[0]
    error = compute_error(y, tx, w)
    MSE = np.power(error, 2).mean() / 2
    return MSE


def compute_gradient(y, tx, w):
    """Computes the gradient at w.
    Returns:
        a loss"""
    error = compute_error(y, tx, w)
    N = y.shape[0]
    grad = -1/N * (np.transpose(tx) @ error) 
    return grad


def log_reg_loss(y, tx, w):
    """compute the cost by negative log likelihood.
    Returns:
        a loss
    """
    ln = np.log(1+np.exp(tx@w))
    mul = y*(tx@w)
    L = (1/tx.shape[0])*(ln-mul)
    loss = np.sum(L)
    return loss


def log_reg_grad(y, tx, w):
    """compute the gradient of loss.
    Returns:
        grad vector
    """
    sig = sigmoid(tx@w)
    grad = (1/tx.shape[0])*np.transpose(tx)@(sig-y)
    return grad


def calculate_hessian(y, tx, w):
    """calculate the Hessian of the loss function.
    Returns:
        hessian matrix
    """
    S = np.eye(tx.shape[0])
    for i in range(tx.shape[0]):
        sn = sigmoid(tx[i,:]@w)*(1-sigmoid(tx[i,:]@w))
        S[i,i] *= sn
    hess = (1/tx.shape[0])*(np.transpose(tx)@S@tx)
    return hess

            
def sigmoid(t):
    """Compute sigmoid function.
    Returns:
        sigmoid value
    """
    sig = 1/(1+np.exp(-t))
    return sig
