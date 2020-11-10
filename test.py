#! /usr/bin/env python3

import ffnn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from train import mnist
from loss_fcns import squared_loss, cross_entropy_loss
import ffnn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from train import mnist
from loss_fcns import squared_loss, cross_entropy_loss
from activations import leaky_ReLU, softmax


net = ffnn.FFNN([784, 256, 10], [leaky_ReLU, softmax], cross_entropy_loss)
net.weights = np.load("good_models/0/weights.npy", allow_pickle=True)
net.biases = np.load("good_models/0/biases.npy", allow_pickle=True)


X_train, Y_train, X_test, Y_test = mnist()
X_test.shape
loss_samp = list(sorted(zip(X_test.T, cross_entropy_loss.f(Y_test, net.feed_forward(X_test))), key=lambda v: v[1]))

n = 45
print(loss_samp[n][1])
imshow(loss_samp[n][0].reshape(28, 28))
plt.show()
