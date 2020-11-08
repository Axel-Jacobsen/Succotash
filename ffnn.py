"""
Philosophy of this network:
    The goal that I had while writing this was for me to cement my understanding of the basic fully connected feed-forward network.
    My original implementation was quite slow, as it was not taking advantage of numpy vectorization - this version does. You can compare
    the previous version of this file to this one (filename nn.py, commit 9cb3da3ce582e940ed862f95930879c8be1721d1), and see the
    significant training speed differences. I will say, adding vectorization makes the code less readable (and also increases the
    required memory) as I had to pad all vectors and matricies with zeros so each set of data had the same shape (and therefore it could be vectorized).
"""

import numpy as np


np.set_printoptions(suppress=True)
np.random.seed(1337)


class FFNN:
    def __init__(self, layers, hs, cost_fcn):
        assert len(hs) == len(layers) - 1
        self.layers = layers
        self.hs = hs
        self.cost_fcn = cost_fcn
        self.weights, self.biases = self.make_network()

    def __repr__(self):
        return f"<FFNN {self.layers} {self.cost_fcn}>"

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
                bound = np.sqrt(2 / layer_arr[i])
                weight_matrix = bound * np.random.normal(size=(dim,prev_dim))#np.random.uniform(low=-bound, high=bound, size=(dim, prev_dim)).astype(np.float32)
                biases_matrix = bound * np.random.normal(size=(dim,1))#np.random.uniform(low=-bound, high=bound, size=(dim, 1)).astype(np.float32)
            else:
                weight_matrix = np.zeros((dim, prev_dim), dtype=np.float32)
                biases_matrix = np.zeros((dim, 1), dtype=np.float32)

            weights.append(weight_matrix)
            biases.append(biases_matrix)
            prev_dim = dim

        return weights, biases

    def feed_forward(self, xs, training=False):
        """
        Feed-forward through the network, saving the activations and non-linearities
        after each layer for backprop.

        xs has to be of shape (num features, batch_size)
        - x, z, a are all vectors of inputs, outputs, and linear outputs at layers
        """
        zs = []
        activations = []

        activation = xs.astype(np.float32) / 255
        activations.append(activation)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = np.einsum("ij, jb -> ib", W, activation) + b
            zs.append(z)
            activation = self.hs[i].f(z)
            activations.append(activation)

        return (activation, activations, zs) if training else activation

    def back_prop(self, xs, ts):
        """
        xs,ts are lists of vectors (ts are targets for training i.e. true output given input x)
        """
        weight_grads, bias_grads = self.make_network(random=False)
        ys, activations, zs = self.feed_forward(xs, training=True)

        # delta_L = grad cost_fcn(outputs) * activation_fcn.deriv(weighted_output_last_layer)
        # should be hadamard product
        # Also, ys is just activations[-1]
        assert ts.shape == ys.shape
        delta = self.cost_fcn.deriv(ts, ys) * self.hs[-1].deriv(zs[-1])

        batch_weights = np.einsum("ib, jb -> ijb", delta, activations[-2])

        # sum along batch
        bias_grads[-1][:, :] = np.mean(delta, axis=-1).reshape(-1, 1)
        weight_grads[-1][:, :] = np.mean(batch_weights, axis=-1)

        # back propogate through layers
        for l in range(2, len(self.layers)):
            nonlinear_deriv = self.hs[-l].deriv(zs[-l])
            delta = np.dot(self.weights[-l + 1].T, delta) * nonlinear_deriv
            batch_weights = np.einsum("ib, jb -> ijb", delta, activations[-l - 1])

            bias_grads[-l][:, :] = np.mean(delta, axis=-1).reshape(-1, 1)
            weight_grads[-l][:, :] = np.mean(batch_weights, axis=-1)

        for new_b, new_g, self_b, self_g in zip(bias_grads, weight_grads, self.biases, self.weights):
            assert new_b.shape == self_b.shape
            assert new_g.shape == self_g.shape

        return weight_grads, bias_grads

    def learn(self, xs, ys, batchs, batch_size, lr):
        """
        xs/ys is the input/output data, batchs is
        the number of batches to train, batch size is the number of input/outputs to
        use in each batch, and lr is learning rate.

        xs, ys have the input vectors as COLUMNS, so xs shape should be (num_features, batch_size)
        e.g. with MNIST, each image is 28*28=784 features, so xs is (784, 60000)
        since there are 10 classes in mnist, y should be (10, 60000)
        """
        N = 1000
        epoch = 0
        batch_acc_avg = 0
        batch_loss_avg = 0
        samples_before_epoch = 0
        losses = []
        accuracies = []
        for batch in range(batchs):
            # get random indicies from batch
            random_indicies = np.random.choice(xs.shape[1], size=batch_size)
            self.mini_batch(xs[:, random_indicies], ys[:, random_indicies], lr)

            # get labels and predicted outputs
            ts = ys[:, random_indicies]
            ys_out_test = self.feed_forward(xs[:, random_indicies])

            batch_loss = self.cost_fcn.f(ts, ys_out_test).mean()
            batch_accuracy = (np.argmax(ts, axis=0) == np.argmax(ys_out_test, axis=0)).mean()

            batch_loss_avg += batch_loss
            batch_acc_avg += batch_accuracy

            losses.append(batch_loss)
            accuracies.append(batch_accuracy)

            samples_before_epoch += batch_size

            if samples_before_epoch > xs.shape[1]:
                samples_before_epoch = 0
                epoch += 1

            if batch % N == 0:
                print(f"epoch {epoch} batch {batch}: \t train loss {batch_loss_avg / N :.6f} \t batch accuracy {batch_acc_avg / N :.6f}")
                batch_loss_avg = 0
                batch_acc_avg = 0

        return losses, accuracies

    def mini_batch(self, batch_xs, batch_ys, lr):
        """
        batch_xs is the batch of inputs, batch_ys is batch of outputs, lr is learning rate
        """
        weight_grads, bias_grads = self.back_prop(batch_xs, batch_ys)

        self.weights = [
            w - lr * weight_grad for w, weight_grad in zip(self.weights, weight_grads)
        ]
        self.biases = [
            b - lr * bias_grad for b, bias_grad in zip(self.biases, bias_grads)
        ]
