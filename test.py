#! /usr/bin/env python3

import ffnn
import numpy as np
import matplotlib.pyplot as plt

from loss_fcns import squared_loss, cross_entropy_loss
from activations import eLU, ReLU, leaky_ReLU, sigmoid, linear, tanh, softmax


net = ffnn.FFNN([2,2,2], [ReLU, softmax], cross_entropy_loss)
net.weights = np.load("weights.npy")
net.biases = np.load("biases.npy")

p1 = np.asarray([[1],[1]])
p2 = np.asarray([[5],[5]])
p3 = np.asarray([[9],[9]])

print(p1, net.feed_forward(p1), end="\n\n")
print(p2, net.feed_forward(p2), end="\n\n")
print(p3, net.feed_forward(p3), end="\n\n")
