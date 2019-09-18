import nn
import numpy as np
import matplotlib.pyplot as plt
from activations import linear, tanh
from loss_fcns import squared_loss

np.random.seed(42)

if __name__ == '__main__':


    """ FORWARD PASS TEST """
    net = nn.NN([3, 5, 1], [linear] * 2, squared_loss)
    y, test_a, test_z = net.feed_forward(np.ones((3,1)))

    # Checking shapes consistency
    assert np.all(test_z[0]==np.array([1,1,1])) # Are the input vector and the first units the same?
    assert np.all(test_z[1]==test_a[0])         # Are the first affine transformations and hidden units the same?
    assert np.all(test_z[2]==test_a[1])         # Are the output units and the affine transformations the same?


    """ BACKWARD PASS TEST """
    from copy import deepcopy

    def finite_diff_grad(x, net, epsilon=None):
        
        if epsilon == None:
            epsilon = np.finfo(np.float32).eps # Machine epsilon for float 32
        
        NN_weights = deepcopy(net.weights)
        test_a, _, _ = net.feed_forward(x) # We evaluate f(x)
        grads = deepcopy(NN_weights)
        
        for h in range(len(NN_weights)):
            for r in range(NN_weights[h].shape[0]):
                for c in range(NN_weights[h].shape[1]):
                    weights_copy           = deepcopy(NN_weights)
                    weights_copy[h][r,c]  += epsilon
                    net.weights            = weights_copy 
                    test_a_eps, _, _       = net.feed_forward(x)
                    grads[h][r,c]          = (test_a_eps - test_a)/epsilon
        
        return grads

    # Set up neural net
    net = nn.NN([3, 5, 1], [linear] * 2, squared_loss)

    # Backward pass
    # Let input to backwards pass be [[1, 1, 1]], and let that be the output as well
    test_g_w, _ = net.back_prop(np.ones((3,1)), np.array(20))
    # Estimation by finite differences
    test_fdg_w = finite_diff_grad(np.ones((3,1)), net)
    

    for l in range(len(test_g_w)):
        print(f'test_g_w[{l}] = {test_g_w[l]}')
        print(f'test_fdg_w[{l}] = {test_fdg_w[l]}')
        assert np.allclose(test_fdg_w[l], test_g_w[l])


