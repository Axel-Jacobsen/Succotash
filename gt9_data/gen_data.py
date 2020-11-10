#! /usr/bin/env python3

import numpy as np

N_train = 100000
N_test = 30000

X_train = np.random.randint(0, high=10, size=(2, N_train))
Y_train = ((X_train[0, :] + X_train[1, :]) > 9).astype(int).reshape(N_train)
np.save("X_train", X_train)
np.save("Y_train", Y_train)

X_test = np.random.randint(0, high=10, size=(2, N_test))
Y_test = ((X_test[0, :] + X_test[1, :]) > 9).astype(int).reshape(N_test)
np.save("X_test", X_test)
np.save("Y_test", Y_test)
