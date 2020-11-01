import numpy as np


class cross_entropy_loss:
    EPS = 1e-9

    def f(t, y):
        ret = -t * np.log(np.clip(y, cross_entropy_loss.EPS, 1 - cross_entropy_loss.EPS))
        a = ret.sum(axis=0)
        return ret.sum(axis=0)

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
