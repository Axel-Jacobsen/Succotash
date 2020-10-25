"""
Philosophy of this network:
    The goal that I had while writing this was for me to cement my understanding of the basic fully connected feed-forward network.
    My original implementation was quite slow, as it was not taking advantage of numpy vectorization - this version does. You can compare
    the previous version of this file to this one (filename nn.py, commit 9cb3da3ce582e940ed862f95930879c8be1721d1), and see the 
    significant training speed differences. I will say, adding vectorization makes the code less readable (and also increases the
    required memory) as I had to pad all vectors and matricies with zeros so each set of data had the same shape (and therefore it could be vectorized).
"""

import numpy as np


class FFNN:
    def __init__(self, layers, hs, cost_fcn):
        assert len(hs) == len(layers) - 1
        self.layers = layers
        self.hs = hs
        self.cost_fcn = cost_fcn
        self.weights, self.biases = self.make_network()

    def make_network(self, random=True):
        """
        random == False for generating empty weight/bias matricies
        """
        layer_arr = self.layers

        weights = []
        biases = []

        layer_iter = iter(layer_arr)
        prev_dim = layer_iter.__next__()

        for i, dim in enumerate(layer_iter):
            if random:
                bound = 4 * np.sqrt(6) / np.sqrt(layer_arr[i] + layer_arr[i + 1])
                weight_matrix = np.random.uniform(low=-bound, high=bound, size=(prev_dim, dim)).astype(np.float32)
                biases_matrix = np.random.uniform(low=-bound, high=bound, size=(1, dim)).astype(np.float32)
            else:
                weight_matrix = np.zeros((prev_dim, dim), dtype=np.float32)
                biases_matrix = np.zeros((dim, 1), dtype=np.float32)

            weights.append(weight_matrix)
            biases.append(biases_matrix)
            prev_dim = dim

        return weights, biases

    def feed_forward(self, xs, training=False):
        """
        Feed-forward through the network, saving the activations and non-linearities
        after each layer for backprop.

        xs has to be of shape (batch_size, num features)
        - x, z, a are all vectors of inputs, outputs, and linear outputs at layers
        """
        batch_size, num_features = xs.shape

        # make activations and zs a uniform size; that way we can do vectorization
        # zs direct output from layer
        # ignore the first layer, as this is the input layer.
        zs = [np.zeros((batch_size, layer_width), dtype=np.float32) for layer_width in self.layers[1:]]
        # activations are zs after nonlinearities
        activations = [np.zeros((batch_size, layer_width), dtype=np.float32) for layer_width in self.layers]

        activation = xs.astype(np.float32)
        activations[0][:, :] = activation
        for i, W in enumerate(self.weights):
            z = np.dot(activation, W) + self.biases[i]
            zs[i] = z
            activation = self.hs[i].f(z)
            activations[i + 1] = activation

        y = activations[-1]
        return y, activations, zs if training else y

    def back_prop(self, xs, ts, batch=False):
        """
        xs,ts are lists of vectors (ts are targets for training i.e. true output given input x)
        """
        grads, biases = self.make_network(random=False)
        ys, activations, zs = self.feed_forward(xs, training=True)

        # delta_L: derivative of Cost fcn w.r.t. zs times derivative of nonlinear fcn of final layer
        delta = self.cost_fcn.deriv(ts, ys) * self.hs[-1].deriv(zs[-1])

        # dC/dw_jk = activation_k * delta_j
        batch_weights = np.dot(zs[-2].T, delta)

        grads[-1] = np.sum(batch_weights, axis=0)
        biases[-1] = np.sum(delta, axis=0)

        # back propogate through layers
        for l in range(2, len(self.layers)):
            nonlinear_deriv = self.hs[-l].deriv(zs[-l])
            delta = np.dot(delta, self.weights[-l + 1].T) * nonlinear_deriv
            batch_weights = np.dot(activations[-l - 1].T, delta)

            biases[-l] = np.sum(delta, axis=0)
            grads[-l] = np.sum(batch_weights, axis=0)

        return grads, biases

    def learn(self, xs, ys, epochs, batch_size, lr):
        """
        xs/ys is the input/output data, xs_val/ys_val is input/output validation, epochs is
        the number of batches to train, batch size is the number of input/outputs to
        use in each batch, and lr is learning rate
        """
        for epoch in range(epochs):
            random_indicies = np.random.choice(xs.shape[0], size=batch_size)
            self.mini_batch(xs[random_indicies, :], ys[random_indicies, :], lr)

            ys_out_test, _, _ = self.feed_forward(xs)

            train_loss = np.mean(self.cost_fcn.f(ys, ys_out_test[:, :1]))

            if epoch % 500 == 0:
                print(f"epoch {epoch} \t train loss {train_loss:.3f}")

    def mini_batch(self, batch_xs, batch_ys, lr):
        """
        batch_xs is the batch of inputs, batch_ys is batch of outputs, lr is learning rate
        """
        weight_grads, bias_grads = self.back_prop(batch_xs, batch_ys)

        self.weights = [
            w - (lr / batch_xs.shape[0]) * weight_grad for w, weight_grad in zip(self.weights, weight_grads)
        ]
        self.biases = [b - (lr / batch_xs.shape[0]) * bias_grad for b, bias_grad in zip(self.biases, bias_grads)]
