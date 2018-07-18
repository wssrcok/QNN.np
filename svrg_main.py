import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from tqdm import tqdm
import copy
from utils import load_dataset, randomize_batch
from model_seq import MNIST_model, MNIST_model_b, \
                      svhn_model, svhn_model_b, \
                      cifar10_model, cifar10_model_b
from layers import reduce_mean_softmax_cross_entropy_loss, \
                   initialize_weights_xavier, update_weights
from quantize import truncate_grads # TEST
from svrg_helper import get_grads, update_weights_SVRG

def plot_costs(costs, learning_rate):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('batchs (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

def save_weights(weights):
    print('save weights[y/n]?')
    if (input() == 'y'):
        print('saving...')
        np.save('./weights', weights)

def load_weights():
    print('loading weights')
    return np.load('./weights.npy').item()

def predict(model, eval_set, weights):
    eval_data, y = eval_set
    y_hat, _ = model(eval_data[0:1024], weights)
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
          num_epochs = 10,
          print_cost = True):
    train_data, train_labels = train_set
    #numpy is really slow, so use 1k example instead of 50k originally
    train_data = train_data[0:256].astype(np.float32)
    train_labels = train_labels[:,0:256]

    costs = []            # keep track of cost
    conv_dim, dense_dim = layer_dims
    
    # initialize weights
    try:
        w_bar = load_weights()
    except IOError:
        print('loading failed, initialize new weights')
        w_bar = initialize_weights_xavier(conv_dim, dense_dim, seed = 1)
    T = batch_size
    for k in tqdm(range(num_epochs)):
        g, cost = get_grads(train_data, train_labels, models, w_bar)
        print ("Cost after Epoch %i: %f" %(k+1, cost))
        w = copy.deepcopy(w_bar)
        for t in tqdm(range(T)):
            idx = np.random.randint(train_data.shape[0])
            dw, loss = get_grads(train_data[idx:idx+1], train_labels[:,idx:idx+1], models, w)
            dw_bar, _ = get_grads(train_data[idx:idx+1], train_labels[:,idx:idx+1], models, w_bar)
            print ("loss after Epoch %i, batch %i: %f" %(k+1, t+1, loss))
            w = update_weights_SVRG(w, g, dw, dw_bar, np.float32(learning_rate))
            if loss < 3:
                    costs.append(loss)
        w_bar = copy.deepcopy(w)
    plot_costs(costs, learning_rate)
    return w_bar

def main():
    args = len(sys.argv)
    if args == 0 or args == 2 or args == 3:
        print('usage1: $ python main.py')
        print('usage2: $ python main.py <batch_size(int)> <learning_rate(float)> <num_epochs(int)>')
        print('usage3: $ python main.py <batch_size(int)> <learning_rate(float)> <num_epochs(int)> <quantize_bits(int)>')
        print('usage4: $ python main.py <batch_size(int)> <learning_rate(float)> <num_epochs(int)> <quantize_bits(int)> <quantize_grads(int)>')
        return None

    data = 'svhn'
    train_data, train_labels, eval_data, eval_labels, classes = load_dataset(data = data)

    train_set = (train_data, train_labels)

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
        
    if data == 'MNIST':
        conv_dims = [(32,1,5,5),(64,32,5,5)] #these two are for MNIST
        dense_dims = [3136, 1024, classes]
        models = (MNIST_model(truncate_f), MNIST_model_b(truncate_b))
    elif data == 'svhn':
        conv_dims = [(32,3,5,5),(64,32,5,5)] #these two are for SVHN
        dense_dims = [4096, 1024, classes]
        models = (svhn_model(truncate_f), svhn_model_b(truncate_b))
    elif data == 'cifar10':
        conv_dims = [(32,3,3,3),(32,32,3,3),(64,32,3,3),(64,64,3,3)]
        dense_dims = [4096, 512, classes]
        models = (cifar10_model(truncate_f), cifar10_model_b(truncate_b))

    layer_dims = (conv_dims, dense_dims)

    weights = train(models, layer_dims, train_set,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    num_epochs=num_epochs,
                    truncate_f=truncate_f,
                    truncate_b=truncate_b)

    eval_set = (eval_data, eval_labels)
    if data == 'MNIST' or data == 'svhn':
        predict(MNIST_model(), eval_set, weights)
    elif data == 'cifar10':
        predict(cifar10_model(), eval_set, weights)

    save_weights(weights)

if __name__ == '__main__':
    main()