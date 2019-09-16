import numpy as np

class cross_entropy_loss:

    def f(t, y):
        return -1 * np.einsum('ij,ij', t, np.log(y))
    
    def deriv(t, y):
         return y - t

class squared_error:

    def f(t, y):
        return 0.5 * (y-t)**2

    def deriv(t, y):
        return y - t

