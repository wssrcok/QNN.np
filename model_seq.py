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

def MNIST_model(truncate = False):
    model = Sequential([
        conv2d(weight_id = 1),
        Quantized_ReLu(truncate = truncate),
        max_pool(),
        conv2d(weight_id = 2),
        Quantized_ReLu(truncate = truncate),
        max_pool(),
        flatten(),
        dense(weight_id = 3),
        Quantized_ReLu(truncate = truncate),
        dense(weight_id = 4),
        softmax()
    ])
    return model

def MNIST_model_b(truncate = False):
    model_b = Sequential_b([
        softmax_b(),
        Quantized_dense_b(grad_id = 4, truncate = truncate),
        ReLu_b(),
        Quantized_dense_b(grad_id = 3, truncate = truncate),
        unflatten(),
        max_pool_b(),
        ReLu_b(),
        Quantized_conv2d_b(grad_id = 2, truncate = truncate),
        max_pool_b(),
        ReLu_b(),
        Quantized_conv2d_b(grad_id = 1, truncate = truncate)
    ])
    return model_b

def cifar10_model(truncate = False):
    model = Sequential([
        conv2d(weight_id = 1),
        Quantized_ReLu(truncate = truncate),
        conv2d(weight_id = 2),
        Quantized_ReLu(truncate = truncate),
        max_pool(),
        dropout(0.25),

        conv2d(weight_id = 3),
        Quantized_ReLu(truncate = truncate),
        conv2d(weight_id = 4),
        Quantized_ReLu(truncate = truncate),
        max_pool(),
        dropout(0.25),

        flatten(),
        dense(weight_id = 5),
        Quantized_ReLu(truncate = truncate),
        dropout(0.5),
        dense(weight_id = 6),
        softmax()
    ])
    return model

def cifar10_model_b(truncate = False):
    model_b = Sequential_b([
        softmax_b(),
        Quantized_dense_b(grad_id = 6, truncate = truncate),
        dropout_b(),
        ReLu_b(),
        Quantized_dense_b(grad_id = 5, truncate = truncate),
        unflatten(),

        dropout_b(),
        max_pool_b(),
        ReLu_b(),
        Quantized_conv2d_b(grad_id = 4, truncate = truncate),
        ReLu_b(),
        Quantized_conv2d_b(grad_id = 3, truncate = truncate),
        
        dropout_b(),
        max_pool_b(),
        ReLu_b(),
        Quantized_conv2d_b(grad_id = 2, truncate = truncate),
        ReLu_b(),
        Quantized_conv2d_b(grad_id = 1, truncate = truncate)
    ])
    return model_b

def svhn_model(truncate = False):
    model = Sequential([
        conv2d(weight_id = 1),
        Quantized_ReLu(truncate = truncate),
        max_pool(),
        dropout(0.5),
        conv2d(weight_id = 2),
        Quantized_ReLu(truncate = truncate),
        max_pool(),
        flatten(),
        dense(weight_id = 3),
        Quantized_ReLu(truncate = truncate),
        dropout(0.5),
        dense(weight_id = 4),
        softmax()
    ])
    return model

def svhn_model_b(truncate = False):
    model_b = Sequential_b([
        softmax_b(),
        Quantized_dense_b(grad_id = 4, truncate = truncate),
        dropout_b(),
        ReLu_b(),
        Quantized_dense_b(grad_id = 3, truncate = truncate),
        unflatten(),
        max_pool_b(),
        ReLu_b(),
        Quantized_conv2d_b(grad_id = 2, truncate = truncate),
        dropout_b(),
        max_pool_b(),
        ReLu_b(),
        Quantized_conv2d_b(grad_id = 1, truncate = truncate)
    ])
    return model_b