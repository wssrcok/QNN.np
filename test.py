import numpy as np

with np.load('./weights.npy') as data:
        weights = data.item()
print(weights)