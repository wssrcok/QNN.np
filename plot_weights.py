import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math
import sys

weights = np.load('./weights.npy').item()
W1 = weights['W1'].reshape(-1)
W2 = weights['W2'].reshape(-1)

if (sys.argv[1] == 'w1'):
	W = W1
elif (sys.argv[1] == 'w2'):
	W = W2

mu, sigma = np.mean(W), math.sqrt(np.var(W))

# the histogram of the data
n, bins, patches = plt.hist(W, 40, normed=1, facecolor='green', alpha=0.75)

# add a 'best fit' line
y = mlab.normpdf(bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=1)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=mu1,\ \sigma=sigma1$')
plt.axis([np.min(W)-0.1, np.max(W)+0.1, 0, 15])
plt.grid(True)

plt.show()