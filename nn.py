#!/usr/bin/env python3

import numpy as np
from functools import reduce
from nonlinears import ReLU
from loss_fcns import cross_entropy_loss

#
# TODO: split loss functions from nonlinear functions
#       

class NN:

    def __init__(self, layers, hs, cost_fcn):
        self.layers = layers
        self.hs = hs
        self.cost_fcn = cost_fcn
        self.weights, self.biases = self.make_network(layers)

    def make_network(self, layer_arr, random=True):
        num_layers = len(layer_arr)
        assert num_layers > 2

        # ndarray holding the weight matricies; the nth element of `weights` is a matrix holding
        # the weights of layer n, where the weight matrix has the weight w_ji in ith column, jth row,
        # connecting neuron j in layer l to neuron i in layer l-1
        # weights[i] are the weights between the ith and i+1th layer
        weights = [] 
        biases  = []

        layer_iter = iter(layer_arr)
        prev_dim = layer_iter.__next__()

        for dim in layer_iter:
            if random:
                weight = np.random.rand(dim, prev_dim)
                bias = np.random.rand(dim, 1)
            else:
                weight = np.zeros((dim, prev_dim))
                bias = np.zeros((dim, 1))

            weights.append(weight)
            biases.append(bias)
            prev_dim = dim
        print(f'len(weights)={len(weights)}')
        return weights, biases

    def feed_forward(self, x, save_data=False):
        '''
        Feed-forward through the entire network
        What is the best way to apply the non-linearities at each layer?
        the np append is adding the bias in the network
        - x, z, a are all vectors of inputs, outputs, and linear outputs at layers
        '''
        out_dict = {
                'y': None,
                'a': [],
                'z': [x]
            }

        z = np.copy(x)

        for i, W in enumerate(self.weights):
            print(W.shape, z.shape)
            a = np.einsum('ij, jk -> ik', W, z) + self.biases[i]
            z = self.hs[i].f(a)
            if save_data:
                out_dict['a'].append(a)
                out_dict['z'].append(z)

        out_dict['y'] = z
        return out_dict

    def mini_batch(self, batch_xs, batch_ys, lr):
        """
        batch_xs is the batch of inputs, batch_ys is batch of outputs, lr is learning rate
        """
        weights, biases = self.make_network(self.layers, random=False)

        for x, y in zip(batch_xs, batch_ys):
            weight_grads, bias_grads = self.back_prop(x, y)
            weights = [weight + weight_grad for weight, weight_grad in zip(weights, weight_grads)]
            biases  = [bias + bias_grad for bias, bias_grad in zip(biases, bias_grads)]

        self.weights = [w - eta * weight_grad for w, weight_grad in zip(self.weights, weights)]
        self.biases  = [b - eta * bias_grad for b, bias_grad in zip(self.biases, weights)]

    def back_prop(self, x, t):
        """
        xs,ts are lists of vectors (ts are targets for training i.e. true output given input x)
        For now output is softmax only
        """
        grads, biases = self.make_network(self.layers, random=False)

        ff_pass = self.feed_forward(x, save_data=True)
        # delta_L: derivative of Cost fcn w.r.t. zs times derivative of nonlinear fcn of last layer
        delta = self.cost_fcn.deriv(t, ff_pass['y']) * self.hs[-1].deriv(ff_pass['a'][-1])

        grads[-1]  = np.einsum('jo, ko -> jo', delta, ff_pass['z'][-2]) # TODO shape SHOULD be (ff_pass['z'][-2].shape[0], delta.shape[1])
        biases[-1] = delta

        # back propogate through the layers
        for l in range(2, len(self.layers)):
            nonlinear_deriv = self.hs[-l].deriv(ff_pass['a'][-l])
            delta = np.einsum('jk, jo -> ko', self.weights[-l+1], delta) * nonlinear_deriv
            grads[-l] = np.einsum('jo, ko -> jo', delta, ff_pass['z'][-l-1])
            biases[-l] = delta

        return grads, biases


if __name__ == '__main__':
    hs = [ReLU] * 4 
    layers = [1,4,2,1]

    nn = NN(layers, hs, cross_entropy_loss)
    y = nn.feed_forward(np.ones((1,1)))
    xs = np.asarray([np.random.rand(1,1) for _ in range(10)])
    ts = np.asarray([np.random.rand(1,1) for _ in range(10)])

    dWs = nn.back_prop(xs[0], ts[0])
    print(dWs)

