#!/usr/bin/env python3

"""
First guess at creating a neural network; goal is to be able to train basic nn, not sure what it should be able to do. Maybe MNIST? Need to learn NN. What can I do with a RN or a basic classfier?
"""

import numpy as np
from functools import reduce
from sigmoids import tanh, sigmoid, softmax

class NN:

    def __init__(self, hs, layers):
        assert len(hs) == len(layers)
        self.layers = layers
        self.hs = hs
        self.weights = self.make_network(layers)

    def make_network(self, layer_arr):
        num_layers = len(layer_arr)
        assert num_layers > 2

        # ndarray holding the weight matricies; the nth element of `weights` is a matrix holding
        # the weights of layer n, where the weight matrix has the weight w_ji in ith column, jth row,
        # connecting neuron j in layer l to neuron i in layer l-1
        # weights[i] are the weights between the ith and i+1th layer
        weights = [] 

        layer_iter = iter(layer_arr)
        prev_dim = layer_iter.__next__()

        # adding 1 to the previous dim adds the bias to the weights,
        # given outputs to our layers always have the value 1 appended to
        # the end of them
        for dim in layer_iter:
            weights.append(np.random.rand(prev_dim + 1, dim))
            prev_dim = dim
        
        return weights

    def feed_forward(self, x, save_data=False):
        '''
        Feed-forward through the entire network
        What is the best way to apply the non-linearities at each layer?
        the np append is adding the bias in the network
        - x, z, a are all vectors of inputs, outputs, and linear outputs at layers
        '''
        z = x
        out_dict = {
                'y': None,
                'a': [],
                'z': []
            }
        
        for i, W in enumerate(self.weights):
            z_prev = np.append(z, 1) # Add the bias back in; size of z_prev won't blow up because the matrix mulitplication with W will reduce its size
            a = np.matmul(z_prev, W)
            z = self.hs[i].f(a)
            if save_data:
                out_dict['a'].append(a)
                out_dict['z'].append(z)

        out_dict['y'] = z
        return out_dict

    def back_prop(self, xs, ts):
        """
        xs,ts are lists of vectors (ts are targets for training i.e. true output given input x)
        For now output is softmax only
        """
        # FEED FORWARD AND COLLECT OUTPUTS FROM EACH LAYER #
        # ITERATE OVER ALL DATA #
        # ff_passes[i] is the output from the ith layer of neurons
        ff_passes = []
        output_dim = self.layers[-1]
        delta_L_vec = np.ndarray((1, output_dim))
        for x, t in zip(xs, ts):
            ff_pass = self.feed_forward(x, save_data=True)
            ff_passes.append(ff_pass)
            delta_L_vec += np.matmul(ff_pass['y'], np.concatenate([t]*output_dim)) - output_dim * t
        
        # BACK PROPOGATE AND CALCULATE GRADIENTS #
        # ITERATE OVER ALL LAYERS #
        deltas = [delta_L_vec]
        grads = []

        for i in range(len(self.layers) - 2, 0, -1):
            # I think this line is wrong, havent tested but wrote it drunk on wine and exhausted with fucking life
            print((ff_passes[i]['a']).shape)
            print((np.matmul(deltas[-1], self.weights[i])).shape)
            delta_l = self.hs[i].deriv(ff_passes[i]['a']) * np.matmul(deltas[-1], self.weights[i])
            deltas.append(delta_l)
            grads.append(np.matmul(delta_l, np.concatenate([ff_passes[i - 1]['z']] * self.layers[i])))

        deltas = deltas[::-1]
        grads = grads[::-1]

        return grads


if __name__ == '__main__':
    hs = [sigmoid] * 3 + [softmax]
    layers = [3,4,2,3]

    nn = NN(hs, layers)
    y = nn.feed_forward(np.ones((1,3)))
    xs = np.asarray([np.random.rand(1,3) for _ in range(10)])
    ts = np.asarray([np.random.rand(1,3) for _ in range(10)])

    dWs = nn.back_prop(xs, ts)
    print(dWs)

