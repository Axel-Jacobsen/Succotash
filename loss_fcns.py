import numpy as np


class cross_entropy_loss:
    def f(t, y):
        return -1 * np.einsum("ij,ij", t, np.log(y))

    def deriv(t, y):
        return y - t


class squared_loss:
    def f(t, y):
        return np.square(y - t) / 2

    def deriv(t, y):
        return y - t


class abs_loss:
    def f(t, y):
        return np.abs(y - t)

    def deriv(t, y):
        return np.greater(y - t, 0).astype(float)
