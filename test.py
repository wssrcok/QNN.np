# pseudo code
intialize_weights()
T = update_interval
n = num_examples
for k in range(1, epochs+1):
	g_k = use w≈ to find average gradient over all data
	w_prev = w≈
	for t in range(1, T + 1):
		idx = np.random.randint(n)
		dw_prev = use w_prev to find gradient over data[idx]
		dw≈ = use w≈ to find gradient over data[idx]
		w = w_prev - learning_rate * (dw_prev - dw≈ + g_k)
