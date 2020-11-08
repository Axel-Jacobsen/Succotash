import numpy as np


class softmax:
    def f(v):
        v_exp = np.exp(v - v.max(axis=0))
        ret_val = v_exp / v_exp.sum(axis=0)
        return ret_val

    def deriv(v):
        return softmax.f(v) * (1 - softmax.f(v))


class sigmoid:
    def f(v):
        return 1 / (1 + np.exp(-1 * v))

    def deriv(v):
        return np.exp(-1 * v) / (1 + np.exp(-1 * v) ** 2)


class tanh:
    def f(v):
        return np.tanh(v)

    def deriv(v):
        return 1 / np.cosh(v) ** 2


class linear:
    def f(v):
        return v

    def deriv(v):
        return np.ones_like(v)


class eLU:
    def f(v):
        return np.greater(v, 0) * v + (1 - np.greater(v, 0)) * (np.exp(v) - 1)

    def deriv(v):
        return np.greater(v, 0) + np.maximum(v, 0) * v + (1 - np.greater(v, 0)) * np.exp(v)


class ReLU:
    def f(v):
        return np.maximum(v, 0)

    def deriv(v):
        return np.greater(v, 0).astype(float)


class leaky_ReLU:
    def f(v):
        return np.maximum(v, 0) + 0.1 * np.minimum(v, 0)

    def deriv(v):
        return np.greater(v, 0).astype(float) + 0.1 * np.less(v, 0).astype(float)
