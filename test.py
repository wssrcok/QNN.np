import numpy as np
import warnings
np.set_printoptions(threshold=np.nan)
weights = np.load('./wtf2.npy')
#a = np.sqrt(weights.reshape(-1))
for i in range(10240):
	print(weights.reshape(-1)[i])
	#a = np.sqrt(weights.reshape(-1)[i])
# warnings.filterwarnings('error')
# for i in weights.reshape(-1):
# 	try:
# 		a = np.sqrt(i)
# 	except RuntimeWarning:
# 		print(i)