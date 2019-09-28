import numpy as np

class NN:

    def __init__(self, layers, hs, cost_fcn):
        self.layers = layers
        self.hs = hs
        self.cost_fcn = cost_fcn
        self.max_row, self.max_col = self.get_weights_matrix_max_shape()
        self.weights, self.biases = self.make_network()

    def get_weights_matrix_max_shape(self):
        max_row = max_col = 0
        for dim, prev_dim in zip(self.layers[1:], self.layers):
            max_row, max_col = max(max_row, dim), max(max_col, prev_dim)
        return max_row, max_col

    def pad_last_2_dims(self, M, bottom_pad, right_pad):
        out = [(0,0) for _ in range(len(M.shape) - 2)]
        out.append((0, bottom_pad))
        out.append((0, right_pad))
        return tuple(out)
    
    def pad_edges(self, M, bottom_pad, right_pad):
        pad_tuple = self.pad_last_2_dims(M, bottom_pad, right_pad)
        return np.pad(M, pad_tuple, 'constant', constant_values=0)

    def make_network(self, random=True):
        """ We padd the weight matricies to the largest weight matrix, so we can vectorize everything and be quick - maybe it will work
        """
        layer_arr = self.layers
        max_row, max_col = self.max_row, self.max_col
        assert len(layer_arr) > 2

        weights = [] 
        biases  = []

        layer_iter = iter(layer_arr)
        prev_dim = layer_iter.__next__()

        for i, dim in enumerate(layer_iter):
            if random:
                bound = 4 * np.sqrt(6) / np.sqrt(layer_arr[i] + layer_arr[i+1])
                weight = np.random.uniform(low=-bound, high=bound, size=(dim, prev_dim))
                bias = np.random.uniform(low=-bound, high=bound, size=(dim, 1))
                padded_weights = self.pad_edges(weight, max_row - dim, max_col - prev_dim)
                padded_biases  = self.pad_edges(bias,   max_row - dim, 0)
            else:
                padded_weights = np.zeros((max_row, max_col))
                padded_biases  = np.zeros((max_row, 1))

            weights.append(padded_weights)
            biases.append(padded_biases)
            prev_dim = dim
        
        return weights, biases

    def feed_forward(self, xs):
        """
        We can only take 1D data rn
        Feed-forward through the entire network
        xs has to be of shape (batch_size, num_rows, 1)
        - x, z, a are all vectors of inputs, outputs, and linear outputs at layers
        """
        col_xs = np.copy(xs)
        batch_size, num_rows, num_cols = col_xs.shape
        z = self.pad_edges(col_xs, self.max_row - num_rows, 0)
        ays = np.zeros((batch_size, len(self.layers) - 1, num_rows, num_cols))
        zs = np.zeros((batch_size, len(self.layers), num_rows, num_cols))
        for i, W in enumerate(self.weights):
            a = np.einsum('ij, bjk -> bik', W, zs[-1]) + self.biases[i] 
            ays[:, i, :, :] = a
            zs[:, i + 1, :, :] = self.hs[i].f(a)
 
        y = np.asarray(zs[:, -1, :, :])
        return y, ays, zs

    def learn(self, xs, ys, xs_val, ys_val, epochs, batch_size, lr):
        """
        xs/ys is the input/output data, xs_val/ys_val is input/output validation, epochs is the number of batches to train, batch size
        is the number of input/outputs to use in each batch, and lr is learning rate
        """
        for epoch in range(epochs):
            random_indicies = np.random.choice(xs.shape[0], size=batch_size)
            self.mini_batch(xs[random_indicies, :], ys[random_indicies, :], lr)

            ys_out_test, _,_ = self.feed_forward(xs, batch=True)
            ys_out_val, _, _ = self.feed_forward(xs_val, batch=True)
            train_loss = np.mean(self.cost_fcn.f(ys, ys_out_test))
            val_loss = np.mean(self.cost_fcn.f(ys_val, ys_out_val))

            if epoch % 500 == 0:
                print(f'epoch {epoch} \t val accuracy {val_loss:.3f} \t train accuracy {train_loss:.3f}')

    def mini_batch(self, batch_xs, batch_ys, lr):
        """
        batch_xs is the batch of inputs, batch_ys is batch of outputs, lr is learning rate
        """
        weights, biases = self.make_network(self.layers, random=False)

        weight_grads, bias_grads = self.back_prop(batch_xs, batch_ys, batch=True)
        weights = [weight + weight_grad for weight, weight_grad in zip(weights, weight_grads)]
        biases  = [bias + bias_grad for bias, bias_grad in zip(biases, bias_grads)]

        self.weights = [w - lr * weight_grad for w, weight_grad in zip(self.weights, weights)]
        self.biases  = [b - lr * bias_grad for b, bias_grad in zip(self.biases, biases)]

    def back_prop(self, xs, ts, batch=False):
        """
        xs,ts are lists of vectors (ts are targets for training i.e. true output given input x)
        """
        grads, biases = self.make_network(self.layers, random=False)
        ys, ays, zs = self.feed_forward(xs)
        # delta_L: derivative of Cost fcn w.r.t. zs times derivative of nonlinear fcn of final layer
        delta = self.cost_fcn.deriv(ts, ys) * self.hs[-1].deriv(ays[: -1, :, :])
        """ dC/dw_jk = a_k * d_j """
        grads[-1]  = np.einsum('bko, bjo -> bjk', zs[:, -2, :, :], delta) 
        biases[-1] = delta
        # back propogate through the layers
        for l in range(2, len(self.layers)):
            nonlinear_deriv = self.hs[-l].deriv(ays[:, -l, :, :])
            delta = np.einsum('jk, bjo -> bko', self.weights[-l+1], delta) * nonlinear_deriv
            grads[-l] = np.einsum('bjo, bko -> kj', zs[:, -l-1, :, :], delta)
            biases[-l] = np.einsum('bko, bko -> ko', delta, delta)

        return grads, biases

