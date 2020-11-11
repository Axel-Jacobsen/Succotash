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
loss_samp = sorted(list(zip(X_test.T, cross_entropy_loss.f(Y_test, net.feed_forward(X_test)))), key=lambda v: v[1])
G = 10
worst = np.asarray([v[0] for v in loss_samp[-G*G:]])
best = np.asarray([v[0] for v in loss_samp[:G*G]])
imshow(np.concatenate(worst.reshape(G, 28*G, 28), axis=1))
# imshow(np.concatenate(best.reshape(G, 28*G, 28), axis=1))
plt.axis('off')
plt.show()
