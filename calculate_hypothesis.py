import numpy as np
from sigmoid import *


def calculate_hypothesis(X, theta, i):
    """
        :param X            : 2D array of our dataset
        :param theta        : 1D array of the trainable parameters
        :param i            : scalar, index of current training sample's row
    """
    #hypothesis = 0.0
    #########################################
    # You must calculate the hypothesis for the i-th sample of X, given X, theta and i.
    # since we have two features used to classify to the binary classes y=0 and y=1
    hypothesis = np.dot(theta, X[i])
    # /
    # hypothesis = theta[0] + X[i, 1] * theta[1] + X[i, 2] * theta[2]
    # hypothesis = np.matmul(theta, X[i] )
    result = sigmoid(hypothesis)

    return result
