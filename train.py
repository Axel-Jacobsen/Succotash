#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plta

import ffnn

from loss_fcns import squared_loss, cross_entropy_loss
from activations import ReLU, leaky_ReLU, sigmoid, linear, tanh


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
        x_train[16:].reshape((-1, 28, 28)),
        y_train[8:].reshape((-1, 1)),
        x_test[16:].reshape((-1, 28, 28)),
        y_test[8:].reshape((-1, 1)),
    )


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = mnist()
    net = ffnn.FFNN([784, 128, 10], [leaky_ReLU, leaky_ReLU, log_softmax], cross_entropy_loss)
    net.learn(X_train.reshape((-1, 28 * 28)), Y_train, 10000, 32, 1e-3)

    print("Test loss: {:.3f}".format(np.mean(cross_entropY_loss.f(Y_test, net.feed_forward(X_test)[0][:, :1]))))
    plt.scatter(X_test.reshape((-1, 28 * 28)), Y_test, label="true")
    plt.scatter(X_test.reshape((-1, 28 * 28)), net.feed_forward(X_test)[0][:, :1], label="net")
    plt.legend()
    plt.show()
