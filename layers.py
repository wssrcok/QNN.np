import numpy as np
from im2col import *
from quantize import truncate_features, truncate_weights

def conv2d(weight_id = -1, hparameters = {'stride': 1, 'pad': 2}, truncate = 0):
    def layer(A_prev, parameters):
        print('conv2d',  end=' => ', flush=True)
        W = parameters["W"+str(weight_id)]
        b = parameters["b"+str(weight_id)]
        n_C, n_C_prev, f, f = W.shape
        m, n_C_prev, n_H_prev, n_W_prev = A_prev.shape

        stride = hparameters['stride']
        pad = hparameters['pad']
        A_prev_col = im2col_indices(A_prev, f, f, padding=pad, stride=stride)
        W_col = W.reshape(n_C, -1)

        out = W_col @ A_prev_col + b
        n_H_shape = (n_H_prev - f + 2 * pad) / stride + 1
        n_W_shape = (n_W_prev - f + 2 * pad) / stride + 1
        n_H_shape, n_W_shape = int(n_H_shape), int(n_W_shape)

        out = out.reshape(n_C, n_H_shape, n_W_shape, m)
        out = out.transpose(3, 0, 1, 2)
        # now Z.shape is (m, n_C, n_H_shape, n_W_shape)

        cache = (A_prev, W, b, hparameters, A_prev_col)
        return out, cache
    return layer

def conv2d_b(grad_id = -1):
    '''
    Arguments:
    dZ -- numpy array of shape (m, n_C, n_H, n_W) 
    cache -- cache of values needed for the conv_backward(), output of conv_forward()

    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_C_prev, n_H_prev, n_W_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (n_C_prev, n_C, f, f)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (n_C, 1)
    '''
    def layer(dZ, cache, grads):
        print('conv2d_b',  end=' => ', flush=True)
        A_prev, W, b, hparameters, A_prev_col = cache
        n_C, n_C_prev, f, f = W.shape 

        db = np.sum(dZ, axis=(0, 2, 3))
        db = db.reshape(n_C, -1)
        dZ_reshaped = dZ.transpose(1, 2, 3, 0).reshape(n_C, -1)
        dW = dZ_reshaped @ A_prev_col.T
        dW = dW.reshape(W.shape)

        stride = hparameters['stride']
        pad = hparameters['pad']

        W_reshape = W.reshape(n_C, -1)
        dA_prev_col = W_reshape.T @ dZ_reshaped
        # turn column to image
        dA_prev = col2im_indices(dA_prev_col, A_prev.shape, f, f, padding=pad, stride=stride)
        grads["dW"+str(grad_id)] = dW
        grads["db"+str(grad_id)] = db
        return dA_prev
    return layer
    
def Quantized_ReLu(truncate = False):
    """
    Implement the RELU function.
    Arguments:
    Z -- Output of the linear layer, of any shape
    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    def layer(x, dummy_parameters):
        print('relu',  end=' => ', flush=True)
        out = np.maximum(0,x)
        cache = x
        if truncate:
            out = truncate_features(A, truncate)
        return out, cache
    return layer

def ReLu_b():
    """
    Implement the backward propagation for a single RELU unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    def layer(dA, cache, dummy_grads):
        print('relu_b',  end=' => ', flush=True)
        Z = cache
        dZ = np.array(dA, copy=True) # just converting dz to a correct object.
        # When z <= 0, you should set dz to 0 as well.
        dZ[Z <= 0] = 0
        
        assert (dZ.shape == Z.shape)
        
        return dZ
    return layer

def softmax():
    def layer(Z, dummy_parameters):
        print('softmax')
        Z_exp = np.exp(Z);
        den = np.sum(Z_exp, axis = 0);
        A = Z_exp / den;
        cache = Z
        return A, cache
    return layer

def softmax_b():
    def layer(Y_hat, cache, dummy_grads):
        print('softmax_b',  end=' => ', flush=True)
        Y = cache
        dZ = Y_hat - Y
        return dZ
    return layer

def max_pool(hparameters = {'f': 2, 'stride': 2}):
    def layer(A_prev, dummy_parameters):
        print('max_pool',  end=' => ', flush=True)
        # Let say our input X is 5x10x28x28
        # Our pooling parameter are: size = 2x2, stride = 2, padding = 0
        # i.e. result of 10 filters of 3x3 applied to 5 imgs of 28x28 with stride = 1 and padding = 1

        # First, reshape it to 50x1x28x28 to make im2col arranges it fully in column
        (m, n_C_prev, n_H_prev, n_W_prev) = A_prev.shape
        A_prev_reshaped = A_prev.reshape(m * n_C_prev, 1, n_H_prev, n_W_prev)

        f = hparameters["f"]
        stride = hparameters["stride"]
        # The result will be 4x9800
        # Note if we apply im2col to our 5x10x28x28 input, the result won't be as nice: 40x980
        A_prev_col = im2col_indices(A_prev_reshaped, f, f, padding=0, stride=stride)

        # Next, at each possible patch location, i.e. at each column, we're taking the max index
        max_idx = np.argmax(A_prev_col, axis=0)

        # Finally, we get all the max value at each column
        # The result will be 1x9800
        A = A_prev_col[max_idx, range(max_idx.size)]

        # Reshape to the output size: 14x14x5x10
        A = A.reshape(n_H_prev//f, n_W_prev//f, m, n_C_prev)

        # Transpose to get 5x10x14x14 output
        A = A.transpose(2, 3, 0, 1)
        # now A is shape (m,n_C_prev, n_H, n_W)
        cache = (A_prev, A_prev_col, max_idx, hparameters)
        return A, cache
    return layer

def max_pool_b():
    def layer(dA, cache, dummy_grads):
        print('max_pool_b',  end=' => ', flush=True)
        (A_prev, A_prev_col, max_idx, hparameters) = cache
        dA_prev_col = np.zeros_like(A_prev_col)

        f = hparameters["f"]
        stride = hparameters["stride"]

        #dA.shape is m,n_H,n_W,n_C
        (m, n_C, n_H, n_W) = dA.shape
        # 5x10x14x14 => 14x14x5x10, then flattened to 1x9800
        # Transpose step is necessary to get the correct arrangement
        dA_flat = dA.transpose(2, 3, 0, 1).ravel()

        # Fill the maximum index of each column with the gradient

        # Essentially putting each of the 9800 grads
        # to one of the 4 row in 9800 locations, one at each column
        dA_prev_col[max_idx, range(max_idx.size)] = dA_flat

        # We now have the stretched matrix of 4x9800, then undo it with col2im operation
        # dX would be 50x1x28x28
        dA_prev = col2im_indices(dA_prev_col, (m * n_C, 1, n_H*f, n_W*f), f, f, padding=0, stride=stride)

        dA_prev = dA_prev.reshape(A_prev.shape)

        return dA_prev
    return layer

def flatten():
    def layer(x, dummy_parameters):
        print('flatten',  end=' => ', flush=True)
        cache = x.shape
        out = x.reshape(cache[0],-1).T
        return out, cache
    return layer

def unflatten():
    def layer(x, cache, dummy_grads):
        print('unflatten',  end=' => ', flush=True)
        out = x.T
        a,b,c,d = cache
        out = out.reshape(a,b,c,d)
        return out
    return layer

def dense(weight_id = -1, truncate = 0):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    def layer(A, parameters):
        print('dense',  end=' => ', flush=True)
        W = parameters["W"+str(weight_id)]
        b = parameters["b"+str(weight_id)]
        Z = np.dot(W,A)+b
        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
        
        return Z, cache
    return layer

def dense_b(grad_id = -1):
    def layer(dZ, cache, grads):
        print('dense_b',  end=' => ', flush=True)
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1/m*np.dot(dZ,A_prev.T)
        db = 1/m*np.sum(dZ,axis=1,keepdims=True)
        dA_prev = np.dot(W.T,dZ)
        
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        grads["dW"+str(grad_id)] = dW
        grads["db"+str(grad_id)] = db
        return dA_prev
    return layer

def reduce_mean_softmax_cross_entropy_loss(Y_hat, Y, truncate = False):
    """
    Arguments:
    Y_hat -- probability vector corresponding to your label predictions, shape (10, number of examples)
    Y -- true "label" vector (for example: [1,0,0,...,0] as 0
         [0,1,0,...,0] as 1), shape (classes, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]
    if truncate:
        fc_out = np.where(fc_out == 0,1/256,fc_out) # TODO: clip value may need to be changed.
        fc_out = np.where(fc_out == 1,255/256,fc_out) # TODO: clip value may need to be changed.
    # Compute loss from aL and y.
    cost = (-1/m) * np.sum(Y*np.log(Y_hat))
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    
    return cost

def initialize_weights_random_normal(filter_dims, layer_dims, truncate = 0, seed = 1):
    '''
    Arguments:
    filter_dims -- dimension of filter:[(f,f,n_C_prev, n_C),(f,f,n_C_prev, n_C)]
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    Returns: parameters -- python dictionary containing your parameters
    for MNIST: W1 b1 is weights and bias for conv_layer1, W2 b2 is for conv_ayer2 
               W3 b3 is for dense_layer1, W4 B4 is for dense_layer2
    '''
    np.random.seed(seed)
    parameters = {}
    L = len(filter_dims)
    L2 = len(layer_dims)
    l = 0
    for l in range(L):
        n_C, n_C_prev, f, f = filter_dims[l]
        parameters['W' + str(l+1)] = np.random.randn(n_C, n_C_prev, f, f) * 0.25
        parameters['b' + str(l+1)] = np.zeros((n_C,1))
        if truncate:
            parameters["W" + str(l+1)] = truncate_weights(parameters["W" + str(l+1)], truncate)
        l+=1
    
    for l in range(1, L2):
        
        parameters['W' + str(l+L)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
        parameters['b' + str(l+L)] = np.zeros((layer_dims[l],1))
        if truncate:
            parameters["W" + str(l+L)] = truncate_weights(parameters["W" + str(l+L)], truncate, weights_range = 0.08)
            parameters["b" + str(l+L)] = truncate_weights(parameters["b" + str(l+L)], truncate, weights_range = 0.08)
        l+=1
    return parameters

def update_weights(parameters, grads, learning_rate, truncate = 0):
    '''
    '''
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads['dW'+str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads['db'+str(l+1)]
        if truncate:
            parameters["W" + str(l+1)] = truncate_weights(parameters["W" + str(l+1)], truncate)
            parameters["b" + str(l+1)] = truncate_weights(parameters["b" + str(l+1)], truncate)
    return parameters