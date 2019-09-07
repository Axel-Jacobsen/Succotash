import numpy as np

class sigmoid:

    def f(v):
        return 1 / (1 + np.exp(-1*v))

    def deriv(v):
        return np.exp(-1*v) / (1 + np.exp(-1*v)**2)

class tanh:

    def f(v):
        return np.tanh(v)

    def deriv(v):
        return 1 / np.cosh(v)**2

class softmax:

    # calculate softmax; needs the sum of exponentials of
    # last layer, so v is the value, vs is all of the as
    def f(v: float, vs: np.ndarray):
        return np.exp(v) / (np.sum(np.exp(vs)))



