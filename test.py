#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from matplotlib.pyplot import imshow

import ffnn
from train import mnist
from loss_fcns import squared_loss, cross_entropy_loss
from activations import leaky_ReLU, softmax


net = ffnn.FFNN([784, 256, 10], [leaky_ReLU, softmax], cross_entropy_loss)
net.weights = np.load("good_models/weights.npy", allow_pickle=True)
net.biases = np.load("good_models/biases.npy", allow_pickle=True)


def random_samples(N: int = 10):
    """Show N random samples one at a time"""
    X_train, Y_train, X_test, Y_test = mnist()
    random_indicies = np.random.choice(X_test.shape[1], size=N)

    for i in random_indicies:
        label = Y_test[:, i].argmax()
        ff = net.feed_forward(X_test[:, i : i + 1])
        prediction = ff.argmax()
        confidence = ff[prediction][0]
        plt.title(
            f"Label: {label}, Prediction: {prediction}, Confidence: {confidence:.3f}",
            color="r" if label != prediction else "k",
        )
        imshow(X_test[:, i].reshape(28, 28))
        plt.show()


def get_grid(G: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Get a grid of the best and worst images for the network"""
    X_train, Y_train, X_test, Y_test = mnist()
    # Sort test set by loss
    loss_samp = sorted(
        list(zip(X_test.T, cross_entropy_loss.f(Y_test, net.feed_forward(X_test)))),
        key=lambda v: v[1],
    )
    # Pick just the images of the the top G^2 and bottom G^2 losses
    best = np.asarray([v[0] for v in loss_samp[: G * G]])
    worst = np.asarray([v[0] for v in loss_samp[-G * G :]])
    return (
        np.concatenate(best.reshape(G, 28 * G, 28), axis=1),
        np.concatenate(worst.reshape(G, 28 * G, 28), axis=1),
    )


def get_test_set_accuracy() -> Tuple[float, float]:
    X_train, Y_train, X_test, Y_test = mnist()
    test_out = net.feed_forward(X_test)
    test_argmax = np.argmax(test_out, axis=0)
    Y_test_argmax = np.argmax(Y_test, axis=0)

    test_losses = cross_entropy_loss.f(Y_test, test_out)
    test_accuracy = (Y_test_argmax == test_argmax).sum() / Y_test.shape[1]

    return np.mean(test_losses), test_accuracy


if __name__ == "__main__":
    loss, accuracy = get_test_set_accuracy()
    print(f"Mean Loss: {loss:.5f}, Accuracy: {accuracy:.5f}")

    random_samples(10)

    best, worst = get_grid()

    plt.title("Best")
    imshow(best)
    plt.show()

    plt.title("Worst")
    imshow(worst)
    plt.show()
