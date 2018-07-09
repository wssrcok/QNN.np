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
    #bits = 8
    weights_range = np.max(w) - np.min(w)
    delta = (1<<bits)/weights_range # delta is scaling factor
    q = np.clip(unbiased_rounding(delta * w)/delta, -weights_range/2, weights_range/2-1/delta) # [-1,1)
    return q.astype(np.float32)
    # return w

def truncate_features(f, bits):
    '''
    Argument:
    f -- np array of features
    bits -- how many bits to quantize

    Return:
    q -- quantized weights
    '''
    #bits = 2
    _max = np.ndarray.max(f)
    n = (1<<bits)/_max # scaling factor
    q = np.clip(unbiased_rounding(f*n)/n, 0, _max-1/n) # [0,256)
    return q.astype(np.float32)
    # return f

def truncate_grads(grads):
    for k,v in grads.items():
        grads[k] = truncate_weights(v, 2)
    print('sample grads:' + str(grads['dW1'][0,0]))