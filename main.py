import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from utils import load_dataset, randomize_batch
from model_seq import MNIST_model, MNIST_model_b
from layers import reduce_mean_softmax_cross_entropy_loss, \
                   initialize_weights_random_normal, update_weights

def plot_costs(costs, learning_rate):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('batchs (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

def train(models, layer_dims, train_set, 
          truncate = False, 
          learning_rate = 0.02, 
          decay_rate = 2,
          batch_size = 32, 
          num_epochs = 10):
    train_data, train_labels = train_set
    #numpy is really slow, so use 1k example instead of 50k originally
    train_data = train_data[0:1024]
    train_labels = train_labels[:,0:1024]

    model, model_b = models
    costs = []            # keep track of cost
    grads = {}
    conv_dim, dense_dim = layer_dims
    
    # initialize weights
    weights = initialize_weights_random_normal(conv_dim, dense_dim, seed = 1)

    batchs = train_data.shape[0] // batch_size

    learning_rate_o = learning_rate
    for i in range(num_epochs):
        # batch is randomized
        train_data, train_labels = randomize_batch(train_data, train_labels)
        for j in range(batchs):
            begin = j*batch_size
            end = (j+1)*batch_size
            x = train_data[begin:end]
            y = train_labels[:,begin:end]

            # model goes entire forward pass
            y_hat, caches = model(x, weights)
            caches[-1] = y # y is cached for softmax_b
            cost = reduce_mean_softmax_cross_entropy_loss(y_hat, y, truncate = truncate)
            # model_b goes entire backward pass
            model_b(y_hat, caches, grads) # this will also update all gradients

            # update weights using minibatch gradient decent
            learning_rate = 1 / (1 + decay_rate * i) * learning_rate_o
            update_weights(weights, grads, learning_rate, truncate = truncate)

            # print cost
            print ("\nCost after Epoch %i, batch %i: %f \n" %(i+1, j+1, cost))
            costs.append(cost)
        print('Epoch %i, Done!\n' %(i+1))

    plot_costs(costs, learning_rate)

# TODO: add quantization
def main():
    if len(sys.argv) != 5 and len(sys.argv) != 4 and len(sys.argv) != 1:
        print('usage1: $ python main.py')
        print('usage2: $ python main.py <batch_size(int)> <learning_rate(float)> <num_epochs(int)>')
        print('usage3: $ python main.py <batch_size(int)> <learning_rate(float)> <num_epochs(int)> <quantize_bits(int)>')
        return None

    train_data, train_labels, eval_data, eval_labels, classes = load_dataset()
    train_set = (train_data, train_labels)

    conv_dims = [(32,1,5,5),(64,32,5,5)]
    dense_dims = [3136, 1024, classes]
    layer_dims = (conv_dims, dense_dims)

    models = (MNIST_model, MNIST_model_b) 

    if len(sys.argv) == 1:
        train(models, layer_dims, train_set)
    elif len(sys.argv) == 4:
        train(models, layer_dims, train_set,
            batch_size = int(sys.argv[1]),
            learning_rate = float(sys.argv[2]),
            num_epochs = int(sys.argv[3]))
    else: 
        train(models, layer_dims, train_set,
            batch_size = int(sys.argv[1]),
            learning_rate = float(sys.argv[2]),
            num_epochs = int(sys.argv[3]),
            truncate = int(sys.argv[4]))

if __name__ == '__main__':
    main()