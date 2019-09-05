#!/usr/bin/env python3

"""
First guess at creating a neural network; goal is to be able to train basic nn, not sure what it should be able to do. Maybe MNIST? Need to learn NN. What can I do with a RN or a basic classfier?
"""

import numpy as np
from functools import reduce
from sigmoids import tanh, sigmoid

def make_network(*layer_arr):
    num_layers = len(layer_arr)
    assert num_layers > 2

    # ndarray holding the weight matricies; the nth element of `weights` is a matrix holding
    # the weights of layer n, where the weight matrix has the weight w_ji in ith column, jth row,
    # connecting neuron j in layer l to neuron i in layer l-1
    weights = [] 

    layer_enum = enumerate(layer_arr)
    _, prev_dim = layer_enum.__next__()

    for layer, dim in layer_enum:
        weights.append(np.random.rand(prev_dim, dim))
        prev_dim = dim

    return weights

def feed_forward(x, weights):
    # i think it is janky casting the row to a matrix, but oh well
    # interestingly enough, it changes the row vector of weights to a 
    # column vector
    assert x.shape == np.asmatrix(weights[0][:,0]).shape
    a = x
    for W in weights:
        a = np.matmul(a, W)
        # a = sigmoid.f(a)
    return a

if __name__ == '__main__':
    ws = make_network(3,4,2)
    y = feed_forward(np.ones((1,3)), ws)
    print(y)
