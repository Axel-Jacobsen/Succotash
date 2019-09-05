import numpy as np

class sigmoid:

    def f(v):
        return 1 / (1 + np.exp(-v))

    def deriv(v):
        return np.exp(-v) / (1 + np.exp(-v)**2)

class tanh:

    def f(v):
        return np.tanh(v)

    def deriv(v):
        return 1 / np.cosh(v)**2

class soft_max:

    # calculate softmax; needs the sum of exponentials of
    # last layer, so v is the value, vs is all of the as
    def f(v: float, vs: np.ndarray):
        return np.exp(v) / (np.sum(np.exp(vs)))



