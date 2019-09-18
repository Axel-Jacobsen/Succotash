#!/usr/bin/env python3

import numpy as np

class NN:

    def __init__(self, layers, hs, cost_fcn):
        self.layers = layers
        self.hs = hs
        self.cost_fcn = cost_fcn
        self.weights, self.biases = self.make_network(layers)

    def make_network(self, layer_arr, random=True):
        assert len(layer_arr) > 2

        weights = [] 
        biases  = []

        layer_iter = iter(layer_arr)
        prev_dim = layer_iter.__next__()

        for dim in layer_iter:
            if random:
                weight = np.random.randn(dim, prev_dim)
                bias = np.random.randn(dim, 1)
            else:
                weight = np.zeros((dim, prev_dim))
                bias = np.zeros((dim, 1))

            weights.append(weight)
            biases.append(bias)
            prev_dim = dim
        
        return weights, biases

    def feed_forward(self, x, batch=False):
        """
        We can only take 1D data rn
        Feed-forward through the entire network
        - x, z, a are all vectors of inputs, outputs, and linear outputs at layers
        """
        if batch:
            z = np.copy(x).reshape(x.shape[0], -1, 1)
            einsum_str = 'ij, bjk -> bik'
        else:
            z = np.copy(x).reshape(-1,1)
            einsum_str = 'ij, jk -> ik'

        ays = []
        zs = [z]
        for i, W in enumerate(self.weights):
            a = np.einsum(einsum_str, W, zs[-1]) + self.biases[i] 
            ays.append(a)
            zs.append(self.hs[i].f(a))

        y = zs[-1]
        return y, ays, zs

    def learn(self, xs, ys, xs_val, ys_val, epochs, batch_size, lr):
        """
        xs/ys is the input/output data, xs_val/ys_val is input/output validation, epochs is the number of batches to train, batch size
        is the number of input/outputs to use in each batch, and lr is learning rate
        """
        for epoch in range(epochs):
            random_indicies = np.random.choice(xs.shape[0], size=batch_size)
            self.mini_batch(xs[random_indicies, :], ys[random_indicies, :], lr)

            ys_test, _,_ = self.feed_forward(xs, batch=True)
            ys_val, _, _ = self.feed_forward(xs_val, batch=True)
            train_loss = np.mean(self.cost_fcn.f(ys_test, ys))
            val_loss = np.mean(self.cost_fcn.f(ys_val, ys))

            if epoch % 500 == 0:
                print(f'epoch {epoch} \t val accuracy {val_loss:.3f} \t train accuracy {train_loss:.3f}')

    def mini_batch(self, batch_xs, batch_ys, lr):
        """
        batch_xs is the batch of inputs, batch_ys is batch of outputs, lr is learning rate
        """
        weights, biases = self.make_network(self.layers, random=False)

        for x, y in zip(batch_xs, batch_ys):
            weight_grads, bias_grads = self.back_prop(x, y)
            weights = [weight + weight_grad for weight, weight_grad in zip(weights, weight_grads)]
            biases  = [bias + bias_grad for bias, bias_grad in zip(biases, bias_grads)]

        self.weights = [w - lr * weight_grad for w, weight_grad in zip(self.weights, weights)]
        self.biases  = [b - lr * bias_grad for b, bias_grad in zip(self.biases, biases)]

    def back_prop(self, x, t):
        """
        xs,ts are lists of vectors (ts are targets for training i.e. true output given input x)
        TODO: make this take batches of data
        """
        grads, biases = self.make_network(self.layers, random=False)
        y, ays, zs = self.feed_forward(x)
        # delta_L: derivative of Cost fcn w.r.t. zs times derivative of nonlinear fcn of final layer
        delta = self.cost_fcn.deriv(t, y) * self.hs[-1].deriv(ays[-1])
        """ dC/dw_jk = a_k * d_j """
        grads[-1]  = np.einsum('ko, jo -> jk', zs[-2], delta) 
        biases[-1] = delta
        # back propogate through the layers
        for l in range(2, len(self.layers)):
            nonlinear_deriv = self.hs[-l].deriv(ays[-l])
            delta = np.einsum('jk, jo -> ko', self.weights[-l+1], delta) * nonlinear_deriv
            grads[-l] = np.einsum('ko, jo -> jk', zs[-l-1], delta)
            biases[-l] = delta

        return grads, biases

