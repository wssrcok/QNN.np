from layers import *    

def Sequential(moduleList):
    def model(x, parameters):
        caches = []
        output = x
        for i,m in enumerate(moduleList):
            output, cache = m(output, parameters)
            caches.append(cache)
        return output, caches
    return model

def Sequential_b(moduleList):
    def model(dy, caches, grads):
        output = dy
        for i,m in enumerate(moduleList):
            cache = caches[-i-1]
            output = m(output, cache, grads)
    return model

MNIST_model = Sequential([
    conv2d(weight_id = 1),
    Quantized_ReLu(),
    max_pool(),
    conv2d(weight_id = 2),
    Quantized_ReLu(),
    max_pool(),
    flatten(),
    dense(weight_id = 3),
    Quantized_ReLu(),
    dense(weight_id = 4),
    softmax()
])

MNIST_model_b = Sequential_b([
    softmax_b(),
    dense_b(grad_id = 4),
    ReLu_b(),
    dense_b(grad_id = 3),
    unflatten(),
    max_pool_b(),
    ReLu_b(),
    conv2d_b(grad_id = 2),
    max_pool_b(),
    ReLu_b(),
    conv2d_b(grad_id = 1)
])