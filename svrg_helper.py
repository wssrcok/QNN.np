from layers import reduce_mean_softmax_cross_entropy_loss

def get_grads(x, y, models, weights):
    model, model_b = models
    grads = {}
    y_hat, caches = model(x, weights)
    caches[-1] = y
    cost = reduce_mean_softmax_cross_entropy_loss(y_hat, y)
    model_b(y_hat, caches, grads)
    return grads, cost

def update_weights_SVRG(w, g, dw, dw_bar, learning_rate):
    L = len(w) // 2
    for l in range(L):
        W_l, b_l = "W" + str(l+1), "b" + str(l+1)
        dW_l, db_l = "dW" + str(l+1), "db" + str(l+1)
        varW = dw_bar[dW_l] - g[dW_l]
        varb = dw_bar[db_l] - g[db_l]
        w[W_l] = w[W_l] - learning_rate * (dw[dW_l] - varW)
        w[b_l] = w[b_l] - learning_rate * (dw[db_l] - varb)
        if (l < 2):
            print(' dw', dw[dW_l][:,0,0,0], '\n', 'var', varW[:,0,0,0])
    return w