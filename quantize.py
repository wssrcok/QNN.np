import numpy as np

def truncate_weights(w, bits):
    '''
    Argument:
    w -- np array of weights
    bits -- how many bits to quantize

    Return:
    q -- quantized weights
    '''
    weights_range = np.max(w) - np.min(w)
    n = (1<<bits)/weights_range
    q = np.clip(np.round(n * w)/n, -weights_range/2, weights_range/2-1/n) # [-1,1)
    return q

def truncate_features(f, bits):
    '''
    Argument:
    f -- np array of features
    bits -- how many bits to quantize

    Return:
    q -- quantized weights
    '''
    _max = np.ndarray.max(f)
    n = (1<<bits)/_max
    q = np.clip(np.round(f*n)/n, 0, _max-1/n) # [0,256)
    return q