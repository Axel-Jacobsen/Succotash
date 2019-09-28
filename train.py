#!/usr/bin/env python3

import nn
import numpy as np
import matplotlib.pyplot as plt
from activations import ReLU, sigmoid, linear
from loss_fcns import squared_loss 

def data_generator(noise=0.1, n_samples=300):
    X = np.linspace(-3, 3, num=n_samples).reshape(-1,1) # 1-D
    np.random.shuffle(X)
    y = np.random.normal((0.5*np.sin(X[:,0]*3) + X[:,0]), noise) # 1-D with trend

    # Stack them together vertically to split data set
    data_set = np.vstack((X.T,y)).T

    train, validation, test = np.split(data_set, [int(0.35*n_samples), int(0.7*n_samples)], axis=0)

    # Standardization of the data, remember we do the standardization with the training set mean and standard deviation
    train_mu = np.mean(train, axis=0)
    train_sigma = np.std(train, axis=0)

    train = (train-train_mu)/train_sigma
    validation = (validation-train_mu)/train_sigma
    test = (test-train_mu)/train_sigma

    x_train, x_validation, x_test = train[:,:-1], validation[:,:-1], test[:,:-1]
    y_train, y_validation, y_test = train[:,-1], validation[:,-1], test[:,-1]
    
    return x_train, y_train.reshape(-1,1),  x_validation, y_validation.reshape(-1,1), x_test, y_test.reshape(-1,1)

if __name__ == '__main__':
    x_train, y_train,  x_validation, y_validation, x_test, y_test = data_generator(noise=0.05, n_samples=1000)
    net = nn.NN([1,8,8,1], [ReLU,sigmoid, linear], squared_loss)
    x_train = x_train.reshape(-1, 1, 1)
    print(x_train)
    ys, ays, zs = net.feed_forward(x_train)

    net.learn(x_train, y_train, x_validation, y_validation, 10000, 64, 1e-3)
    
    print('Final loss:', np.mean(squared_loss.f(y_test, net.feed_forward(x_test, batch=True)[0])))
    plt.scatter(x_test, y_test, label='true')
    plt.scatter(x_test, net.feed_forward(x_test, batch=True)[0], label='net')
    plt.legend()
    plt.show()
