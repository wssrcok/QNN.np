import numpy as np

# def truncate_weights(np_arr, bits, weights_range = 0.5):
#     n = 1<<bits # 256 for 8 bits 512 for 16bits
#     np_arr *= (n/weights_range)
#     np_arr = np.round(np_arr)
#     np_arr = np.where(np_arr >= n/2, n/2-1, np_arr) # >= 128 will become 127
#     np_arr = np.where(np_arr < -n/2, -n/2, np_arr) # < -128 will become -128
#     np_arr /= (n/weights_range)
#     return np_arr

def truncate_weights(w, bits, weights_range = 2):
    n = (1<<bits)/weights_range
    q = np.clip(np.round(n * w)/n, -weights_range/2, weights_range/2-1/n) # [-1,1)
    return q

def truncate_io(np_arr):
    _max = np.ndarray.max(np_arr)
    ratio = 256/_max

    np_arr = np.clip(np.round(np_arr*ratio)/ratio, 0, _max)
    return np_arr