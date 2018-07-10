import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from utils import load_dataset, randomize_batch
from model_seq import MNIST_model, MNIST_model_b
from layers import reduce_mean_softmax_cross_entropy_loss, \
                   initialize_weights_random_normal, update_weights
from quantize import truncate_grads # TEST

def plot_costs(costs, learning_rate):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('batchs (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

def predict(model, eval_set, weights):
    eval_data, y = eval_set
    y_hat, _ = model(eval_data[0:512], weights)
    p = np.zeros(y.shape)
    m = y_hat.shape[1]
    c = y_hat.shape[0]
    print("converting to hard max!")
    for i in range(c):
        for j in range(m):
            if y_hat[i,j] == np.max(y_hat[:,j]):
                p[i,j] = 1
            else:
                p[i,j] = 0
    match = 0
    print("start calculating accuracy!")
    for i in range(m):
        match += int(np.array_equal(p[:,i],y[:,i]))
    print("the acuracy on eval_set is: " + str(match*100/m) + "%")
            

def train(models, layer_dims, train_set, 
          truncate_f = False,
          truncate_b = False, 
          learning_rate = 0.02, 
          decay_rate = 0,
          batch_size = 32, 
          num_epochs = 10):
    train_data, train_labels = train_set
    #numpy is really slow, so use 1k example instead of 50k originally
    train_data = train_data[0:1024].astype(np.float32)
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
            cost = reduce_mean_softmax_cross_entropy_loss(y_hat, y, truncate = truncate_f)
            # model_b goes entire backward pass
            model_b(y_hat, caches, grads) # this will also update all gradients

            truncate_grads(grads, truncate_b) # TESTCODE

            # update weights using minibatch gradient decent
            learning_rate = 1 / (1 + decay_rate * i) * learning_rate_o
            update_weights(weights, grads, np.float32(learning_rate), truncate = truncate_f)

            # print cost
            print ("\nCost after Epoch %i, batch %i: %f \n" %(i+1, j+1, cost))
            if cost < 3:
                costs.append(cost)
        # if i % 2 == 0: # TESTCODE
        #     batch_size *= 2 # TESTCODE batch is starting with 1
        #     batchs = train_data.shape[0] // batch_size # TESTCODE batch is starting with 1
        print('Epoch %i, Done!\n' %(i+1))

    plot_costs(costs, learning_rate)
    return weights

# TODO: add quantization
def main():
    args = len(sys.argv)
    if args == 0 or args == 2 or args == 3:
        print('usage1: $ python main.py')
        print('usage2: $ python main.py <batch_size(int)> <learning_rate(float)> <num_epochs(int)>')
        print('usage3: $ python main.py <batch_size(int)> <learning_rate(float)> <num_epochs(int)> <quantize_bits(int)>')
        print('usage4: $ python main.py <batch_size(int)> <learning_rate(float)> <num_epochs(int)> <quantize_bits(int)> <quantize_grads(int)>')
        return None

    train_data, train_labels, eval_data, eval_labels, classes = load_dataset()
    train_set = (train_data, train_labels)

    conv_dims = [(32,1,5,5),(64,32,5,5)]
    dense_dims = [3136, 1024, classes]
    layer_dims = (conv_dims, dense_dims)

    weights = {}
    batch_size = 32
    learning_rate = 0.02
    num_epochs = 10
    truncate_f = False
    truncate_b = False
    if args >= 4:
        batch_size = int(sys.argv[1])
        learning_rate = float(sys.argv[2])
        num_epochs = int(sys.argv[3])
    if args >= 5:
        truncate_f = int(sys.argv[4])
    if args == 6:
        truncate_b = int(sys.argv[5])
        
    models = (MNIST_model(truncate_f), MNIST_model_b(truncate_b))
    print(truncate_b, truncate_f)
    weights = train(models, layer_dims, train_set,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    num_epochs=num_epochs,
                    truncate_f=truncate_f,
                    truncate_b=truncate_b)

    eval_set = (eval_data, eval_labels)
    predict(MNIST_model(), eval_set, weights)

if __name__ == '__main__':
    main()