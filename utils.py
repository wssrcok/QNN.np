import numpy as np
import tensorflow as tf
import scipy.io as sio

def load_dataset(data = 'MNIST'):
    # Load training and eval data
    if data == 'MNIST':
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        train_data = mnist.train.images # Returns np.array
        classes = 10
        train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        train_labels = one_hot_label(classes, train_labels)
        eval_data = mnist.test.images # Returns np.array
        eval_labels_old = np.asarray(mnist.test.labels, dtype=np.int32)
        eval_labels = one_hot_label(classes, eval_labels_old)

        train_data = train_data.reshape(55000,1,28,28)
        eval_data = eval_data.reshape(10000,1,28,28)

        return train_data, train_labels, eval_data, eval_labels, classes

    elif data == 'cifar10':
        def unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict
        cifar10 = unpickle('cifar10-data/data_batch_1')
        classes = 10
        train_data = cifar10[b'data'].reshape(10000,3,32,32)/256
        train_labels = one_hot_label(classes, np.array(cifar10[b'labels']))
        cifar10_eval = unpickle('cifar10-data/data_batch_2')
        eval_data = cifar10_eval[b'data'].reshape(10000,3,32,32)/256
        eval_labels = one_hot_label(classes, np.array(cifar10_eval[b'labels']))
        
        return train_data, train_labels, eval_data, eval_labels, classes

    elif data == 'svhn':
        svhn = sio.loadmat('svhn-data/train_32x32.mat')
        classes = 10
        train_data = svhn['X'].transpose(3,2,0,1)/256
        train_labels = one_hot_label(classes, svhn['y'].reshape(-1)-1)
        svhn_eval = sio.loadmat('svhn-data/test_32x32.mat')
        eval_data = svhn_eval['X'].transpose(3,2,0,1)/256
        eval_labels = one_hot_label(classes, svhn_eval['y'].reshape(-1)-1)

        return train_data, train_labels, eval_data, eval_labels, classes

def randomize_batch(train_data, train_labels, seed = 1):
    np.random.seed(seed)
    m = train_data.shape[0]
    train_labels = train_labels.T # to make the shape (m, classes)
    permutation = np.random.permutation(m)
    return train_data[permutation], train_labels[permutation].T

def one_hot_label(classes, label):
    """
    reshape label to Sam prefered shape for mnist

    Arguments:
    label -- input label with shape (m,)

    Returns:
    new_label -- output label with shape (classes, m)
    """
    m = label.shape[0]
    new_label = np.zeros((classes, m))
    for i in range(m):
        clas = label[i]
        new_label[clas,i] = 1
    return new_label
