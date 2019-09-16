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

class linear:

    def f(v):
        return v

    def derivative(v):
        return np.ones_like(v)

class ReLU:

    def f(v):
        return np.maximum(v,0)
    
    def deriv(v):
        return (v>0).astype(int)

