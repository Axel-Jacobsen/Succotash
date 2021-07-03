#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import ffnn

from loss_fcns import squared_loss, cross_entropy_loss
from activations import eLU, ReLU, leaky_ReLU, sigmoid, linear, tanh, softmax, MISH


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
        x_train[16:].reshape((28 * 28, -1), order="F"),
        _get_one_hot(y_train[8:], 10).reshape((10, -1)),
        x_test[16:].reshape((28 * 28, -1), order="F"),
        _get_one_hot(y_test[8:], 10).reshape((10, -1)),
    )


if __name__ == "__main__":
    X_train_p, Y_train, X_test_p, Y_test = mnist()

    X_train = X_train_p - np.mean(X_train_p, axis=1)[:, np.newaxis]
    X_test = X_test_p - np.mean(X_test_p, axis=1)[:, np.newaxis]

    print("data loaded")

    net = ffnn.FFNN([784, 256, 10], [MISH, MISH], cross_entropy_loss)

    try:
        losses, accuracies = net.learn(X_train, Y_train, 5000, 128, 1)
    except KeyboardInterrupt:
        pass
    finally:
        plt.plot(range(len(net.losses)), net.losses)
        plt.plot(range(len(net.accuracies)), net.accuracies)
        plt.show()

    np.save("weights.npy", np.asarray(net.weights, dtype=object), allow_pickle=True)
    np.save("biases.npy", np.asarray(net.biases, dtype=object), allow_pickle=True)

    test_out = net.feed_forward(X_test)
    test_argmax = np.argmax(test_out, axis=0)
    Y_test_argmax = np.argmax(Y_test, axis=0)

    test_losses = cross_entropy_loss.f(Y_test, test_out)
    print("Test loss: {:.3f}".format(np.mean(test_losses)))
    print(
        "Test accuracy: {:.3f}".format(
            (Y_test_argmax == test_argmax).sum() / Y_test.shape[1]
        )
    )
