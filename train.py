#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import ffnn

from loss_fcns import squared_loss, cross_entropy_loss
from activations import eLU, ReLU, leaky_ReLU, sigmoid, linear, tanh, softmax


def fake_data():
    def _get_one_hot(targets, num_classes):
        """
        targets (num_samples,)
        output  (num_classes, num_samples)
        """
        ret = np.zeros((num_classes, targets.shape[0]))
        ret[targets, np.arange(targets.size)] = 1
        return ret

    return (
        np.load("fake_data/X_train.npy"),
        _get_one_hot(np.load("fake_data/Y_train.npy"), 2),
        np.load("fake_data/X_test.npy"),
        _get_one_hot(np.load("fake_data/Y_test.npy"), 2),
    )


def mnist():
    def _get_one_hot(targets, num_classes):
        """
        targets (num_samples,)
        output  (num_classes, num_samples)
        """
        ret = np.zeros((num_classes, targets.shape[0]))
        ret[targets, np.arange(targets.size)] = 1
        return ret

    def load_data(fname):
        data_folder = "mnist_data/"
        with open(data_folder + fname, "rb") as f:
            data = f.read()
        return np.frombuffer(data, dtype=np.uint8)

    x_train = load_data("train-images-idx3-ubyte")
    y_train = load_data("train-labels-idx1-ubyte")
    x_test = load_data("t10k-images-idx3-ubyte")
    y_test = load_data("t10k-labels-idx1-ubyte")

    return (
        x_train[16:].reshape((28 * 28, -1), order="C"),
        _get_one_hot(y_train[8:], 10).reshape((10, -1)),
        x_test[16:].reshape((28 * 28, -1), order="C"),
        _get_one_hot(y_test[8:], 10).reshape((10, -1)),
    )


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = fake_data()
    print("data loaded")

    # net = ffnn.FFNN([784, 512, 128, 10], [tanh, tanh, softmax], cross_entropy_loss)
    net = ffnn.FFNN([2, 8, 4, 2], [tanh, tanh, softmax], cross_entropy_loss)

    losses, accuracies = net.learn(X_train, Y_train, 50000, 2*64, 0.001)

    np.save("weights.npy", np.asarray(net.weights, dtype=object))
    np.save("biases.npy", np.asarray(net.biases, dtype=object))

    test_out = net.feed_forward(X_test)
    test_argmax = np.argmax(test_out, axis=0)
    Y_test_argmax = np.argmax(Y_test, axis=0)

    test_losses = cross_entropy_loss.f(Y_test, test_out)
    print("Test loss: {:.3f}".format(np.mean(test_losses)))
    print("Test accuracy: {:.3f}".format(np.sum(Y_test == test_argmax) / Y_test.shape[1]))

    samp_loss = sorted(zip(X_test.T, test_losses), key=lambda v: v[1])

    plt.plot(range(len(losses)), losses)
    plt.plot(range(len(accuracies)), accuracies)
    plt.show()
