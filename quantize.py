import numpy as np

def unbiased_rounding(x):
    '''
    unbiased_rounding
    Argument:
    x -- x must be numpy array.

    Return:
    y -- y is x rounded to nearest integer(unbiased)
    '''
    #generate a random np array with same shape as x
    r = np.random.rand(len(x.reshape(-1))).reshape(x.shape)
    x_decimal = np.remainder(x, 1)
    x[r <= x_decimal] = np.ceil(x[r <= x_decimal])
    x[r > x_decimal] = np.floor(x[r > x_decimal])
    return x

def test_ub_rounding_positive(num = 0.1, k = 1000):
    count1 = 0
    for _ in range(k):
        a = np.array([num])
        if int(unbiased_rounding(a)) == 1:
            count1+=1
    print('the chance of becoming one is ' + str(float(count1)/k))

def test_ub_rounding_negative(num = -0.1, k = 1000):
    count1 = 0
    for _ in range(k):
        a = np.array([num])
        if int(unbiased_rounding(a)) == -1:
            count1+=1
    print('the chance of becoming negative one is ' + str(float(count1)/k))




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
    #q = np.clip(np.round(n * w)/n, -weights_range/2, weights_range/2-1/n) # [-1,1)
    q = np.clip(unbiased_rounding(n * w)/n, -weights_range/2, weights_range/2-1/n) # [-1,1)
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
    #q = np.clip(np.round(f*n)/n, 0, _max-1/n) # [0,256)
    q = np.clip(unbiased_rounding(f*n)/n, 0, _max-1/n) # [0,256)
    return q