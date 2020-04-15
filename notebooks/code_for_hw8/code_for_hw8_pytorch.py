# This code file pertains to problems 3 onwards from homework 8
# Here, we use out-of-the-box neural network framework PyTorch
# to build and train our models

import pdb
import numpy as np
import itertools

import math as m 

import torch
from torch.nn import Linear, ReLU, Conv1d, Conv2d, Flatten, Sequential, CrossEntropyLoss, MSELoss, MaxPool2d, Dropout, MaxPool1d
from torch.optim import Adam
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt

from utils_hw8 import (model_fit, model_evaluate, run_pytorch, call_model, 
                       plot_decision, plot_heat, plot_separator, make_iter,
                       set_weights, set_bias)

######################################################################
# Helper functions for 
# OPTIONAL: Problem 2 - Weight Sharing
######################################################################

def generate_1d_images(nsamples,image_size,prob):
    Xs=[]
    Ys=[]
    for i in range(0,nsamples):
        X=np.random.binomial(1, prob, size=image_size)
        Y=count_objects_1d(X)
        Xs.append(X)
        Ys.append(Y)
    Xs=np.array(Xs)
    Ys=np.array(Ys)
    return Xs,Ys


#count the number of objects in a 1d array
def count_objects_1d(array):
    count=0
    for i in range(len(array)):
        num=array[i]
        if num==0:
            if i==0 or array[i-1]==1:
                count+=1
    return count

def l1_reg(weight_matrix):
    return 0.01 * torch.sum(torch.abs(weight_matrix))    


def filter_reg(weights):
    # Your code here!
    lam = 0
    return lam

def model_reg(model):
    filter_weights = model[0].weight
    return filter_reg(filter_weights)

def get_image_data_1d(tsize,image_size,prob):
    #prob controls the density of white pixels
    #tsize is the size of the training and test sets
    vsize=int(0.2*tsize)
    X_train, Y_train = generate_1d_images(tsize,image_size,prob)
    X_val, Y_val = generate_1d_images(vsize,image_size,prob)
    X_test, Y_test = generate_1d_images(tsize,image_size,prob)
    #reshape the input data for the convolutional layer
    X_train=np.expand_dims(X_train,axis=1)
    X_val=np.expand_dims(X_val,axis=1)
    X_test=np.expand_dims(X_test,axis=1)
    data=(X_train,Y_train,X_val,Y_val,X_test,Y_test)
    return data

def train_neural_counter(layers, data, regularize=False, display=False):
    (X_train, Y_train, X_val, Y_val, X_test, Y_test) = data
    epochs = 10
    batch = 1

    train_iter, val_iter, test_iter = (make_iter(X_train, Y_train),
                                       make_iter(X_val,Y_val),
                                       make_iter(X_test,Y_test))
    model = Sequential(*layers)
    optimizer = Adam(model.parameters())
    criterion = MSELoss()

    model_fit(model, train_iter, epochs, optimizer, criterion, val_iter, 
              history=None, verbose=True, model_reg=model_reg if regularize else None)
    err = model_evaluate(model, test_iter, criterion)
    ws = model[-1].weight
    if display:
        plt.plot(ws)
        plt.show()
    return model,err

######################################################################
# Problem 3
######################################################################

def shifted(X, shift):
    n = X.shape[0]
    m = X.shape[1]
    size = m + shift
    X_sh = np.zeros((n, size, size))
    plt.ion()
    for i in range(n):
        sh1 = np.random.randint(shift)
        sh2 = np.random.randint(shift)
        X_sh[i, sh1:sh1+m, sh2:sh2+m] = X[i, :, :]
        # If you want to see the shifts, uncomment
        #plt.figure(1); plt.imshow(X[i])
        #plt.figure(2); plt.imshow(X_sh[i])
        #plt.show()
        #input('Go?')
    return X_sh

def get_MNIST_data(shift=0):
    train = MNIST(root='./mnist_data', train=True, download=True, transform=None)
    val = MNIST(root='./mnist_data', train=False, download=True, transform=None)
    (X_train, y1), (X_val, y2) = (train.data.numpy(), train.targets.numpy()), \
                                  (val.data.numpy(), val.targets.numpy())
    if shift:
        X_train = shifted(X_train, shift)
        X_val = shifted(X_val, shift)
    return (X_train, y1), (X_val, y2)

# Example Usage:
# train, validation = get_MNIST_data()


def make_deterministic():
    torch.manual_seed(10)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(10)


def weight_reset(l):
    if isinstance(l, Conv2d) or isinstance(l, Linear):
        l.reset_parameters()

def run_pytorch_fc_mnist(train, test, layers, epochs, verbose=True, trials=1, deterministic=True):
    '''
    train, test = input data
    layers = list of PyTorch layers, e.g. [Linear(in_features=784, out_features=10)]
    epochs = number of epochs to run the model for each training trial
    trials = number of evaluation trials, resetting weights before each trial
    '''
    if deterministic:
        make_deterministic()
    (X_train, y1), (X_val, y2) = train, test
    # Flatten the images
    m = X_train.shape[1]
    X_train = X_train.reshape((X_train.shape[0], m * m))
    X_val = X_val.reshape((X_val.shape[0], m * m))

    val_acc, test_acc = 0, 0
    for trial in range(trials):
        # Reset the weights
        for l in layers:
            weight_reset(l)
        # Make Dataset Iterables
        train_iter, val_iter = make_iter(X_train, y1, batch_size=32), make_iter(X_val, y2, batch_size=32)
        # Run the model
        model, vacc, tacc = \
            run_pytorch(train_iter, val_iter, None, layers, epochs, verbose=verbose)
        val_acc += vacc if vacc else 0
        test_acc += tacc if tacc else 0
    if val_acc:
        print("\nAvg. validation accuracy:" + str(val_acc / trials))
    if test_acc:
        print("\nAvg. test accuracy:" + str(test_acc / trials))


def run_pytorch_cnn_mnist(train, test, layers, epochs, verbose=True, trials=1, deterministic=True):
    if deterministic:
        make_deterministic()
    # Load the dataset
    (X_train, y1), (X_val, y2) = train, test
    # Add a final dimension indicating the number of channels (only 1 here)
    m = X_train.shape[1]
    X_train = X_train.reshape((X_train.shape[0], 1, m, m))
    X_val = X_val.reshape((X_val.shape[0], 1, m, m))

    val_acc, test_acc = 0, 0
    for trial in range(trials):
        # Reset the weights
        for l in layers:
            weight_reset(l)
        # Make Dataset Iterables
        train_iter, val_iter = make_iter(X_train, y1, batch_size=32), make_iter(X_val, y2, batch_size=32)
        # Run the model
        model, vacc, tacc = \
            run_pytorch(train_iter, val_iter, None, layers, epochs, verbose=verbose)
        val_acc += vacc if vacc else 0
        test_acc += tacc if tacc else 0
    if val_acc:
        print("\nAvg. validation accuracy:" + str(val_acc / trials))
    if test_acc:
        print("\nAvg. test accuracy:" + str(test_acc / trials))

# Example usage:
# train, validation = get_MNIST_data()
# run_pytorch_fc_mnist(train, validation, [Linear(784, 128), ReLU(), Linear(128, 10)], 1, verbose=True)
# Same pattern applies to the function: run_pytorch_cnn_mnist
