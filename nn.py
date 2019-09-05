#!/usr/bin/env python3

"""
First guess at creating a neural network; goal is to be able to train basic nn, not sure what it should be able to do. Maybe MNIST? Need to learn NN. What can I do with a RN or a basic classfier?
"""

import numpy as np
from functools import reduce
from sigmoids import tanh, sigmoid, softmax

class NN:

    def __init__(self, hs, layer_arr):
        assert len(hs) == len(layer_arr)
        self.make_network(layer_arr)
        self.hs = hs

    def make_network(self, layer_arr):
        num_layers = len(layer_arr)
        assert num_layers > 2

        # ndarray holding the weight matricies; the nth element of `weights` is a matrix holding
        # the weights of layer n, where the weight matrix has the weight w_ji in ith column, jth row,
        # connecting neuron j in layer l to neuron i in layer l-1
        weights = [] 

        layer_iter = iter(layer_arr)
        prev_dim = layer_iter.__next__()

        for dim in layer_iter:
            # adding 1 to the previous dim adds the bias to the weights, 
            # given outputs to our layers always have the value 1 appended to
            # the end of them
            weights.append(np.random.rand(prev_dim + 1, dim))
            prev_dim = dim

        self.weights = weights

    def feed_forward(self, x):
        '''
        Feed-forward through the entire network
        What is the best way to apply the non-linearities at each layer?
        the np append is adding the bias in the network
        '''
        z = x
        for i, W in enumerate(self.weights):
            z = np.append(z, 1)
            a = np.matmul(z, W)
            z = self.hs[i].f(a)
        return a

if __name__ == '__main__':
    hs = [tanh] * 3 + [softmax]
    layers = [3,4,2,1]
    nn = NN(hs, layers)
    y = nn.feed_forward(np.ones((1,3)))
    print(y)
