#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import ffnn

from loss_fcns import squared_loss, cross_entropy_loss
from activations import ReLU, leaky_ReLU, sigmoid, linear, tanh, softmax


def data_generator(noise=0.1, n_samples=300):
    X = np.linspace(-3, 3, num=n_samples).reshape(-1, 1)  # 1-D
    np.random.shuffle(X)
    y = np.random.normal((0.5 * np.sin(X[:, 0] * 3) + X[:, 0]), noise)  # 1-D with trend

    # Stack them together vertically to split data set
    data_set = np.vstack((X.T, y)).T

    train, validation, test = np.split(data_set, [int(0.35 * n_samples), int(0.7 * n_samples)], axis=0)

    # Standardization of the data, remember we do the standardization with the training set mean and standard deviation
    train_mu = np.mean(train, axis=0)
    train_sigma = np.std(train, axis=0)

    train = (train - train_mu) / train_sigma
    validation = (validation - train_mu) / train_sigma
    test = (test - train_mu) / train_sigma

    x_train, x_validation, x_test = (
        train[:, :-1, np.newaxis],
        validation[:, :-1, np.newaxis],
        test[:, :-1, np.newaxis],
    )
    y_train, y_validation, y_test = (
        train[:, -1, np.newaxis],
        validation[:, -1, np.newaxis],
        test[:, -1, np.newaxis],
    )

    return (
        x_train,
        y_train.reshape(-1, 1),
        x_validation,
        y_validation.reshape(-1, 1),
        x_test,
        y_test.reshape(-1, 1),
    )


def mnist():
    def _get_one_hot(targets, nb_classes):
        return np.eye(nb_classes)[np.array(targets).reshape(-1)].reshape((nb_classes, targets.shape[1]))

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
        x_train[16:].reshape((28 * 28, -1)),
        _get_one_hot(y_train[8:].reshape((1, -1)), 10),
        x_test[16:].reshape((28 * 28, -1)),
        _get_one_hot(y_test[8:].reshape((1, -1)), 10),
    )


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = mnist()
    net = ffnn.FFNN([784, 256, 10], [leaky_ReLU, softmax], cross_entropy_loss)
    losses, accuracies = net.learn(X_train, Y_train, 1000, 2, 1e-3)

    np.save("weights.npy", np.asarray(net.weights, dtype=object))
    np.save("biases.npy", np.asarray(net.biases, dtype=object))

    test_out = net.feed_forward(X_test)
    test_argmax = np.argmax(test_out, axis=0)
    Y_test_argmax = np.argmax(Y_test, axis=0)

    print("Test loss: {:.3f}".format(np.mean(cross_entropy_loss.f(Y_test, test_out))))
    print("Test accuracy: {:.3f}".format(np.sum(Y_test_argmax == test_argmax) / Y_test_argmax.shape[0]))

    plt.plot(range(len(losses)), losses)
    plt.plot(range(len(losses)), accuracies)
    plt.show()
