
import sys
import os
import time

import numpy as np
np.random.seed(1234)  # for reproducibility

import theano
import theano.tensor as T

import lasagne

import cPickle
import gzip

import custom_layers

from collections import OrderedDict

if __name__ == "__main__":
    
    # BN parameters
    batch_size = 128
    print("batch_size = "+str(batch_size))
    
    # MLP parameters
    num_units = 1024
    print("num_units = "+str(num_units))
    
    # Decaying LR 
    LR_start = 0.001
    print("LR_start = "+str(LR_start))
    LR_decay = .1
    print("LR_decay = "+str(LR_decay))
    LR_patience = 50
    print("LR_patience = "+str(LR_patience))
    # Training parameters
    # patience = LR_patience * 2.
    patience = 50
    print("patience = "+str(patience))
    
    save_path = "mnist_parameters.npz"
    print("save_path = "+str(save_path))
    
    shuffle_parts = 1
    print("shuffle_parts = "+str(shuffle_parts))
    
    print('Loading MNIST dataset...')
    
    # Loading the MNIST test set
    # You can get mnist.pkl.gz at http://deeplearning.net/data/mnist/mnist.pkl.gz
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    
    train_set_X = train_set[0]
    valid_set_X = valid_set[0]
    test_set_X = test_set[0]
    
    # [0,1] -> [-1,1]
    # train_set_X = 2* train_set_X -1
    # valid_set_X = 2* valid_set_X -1
    # test_set_X = 2* test_set_X -1
    
    # binarizing mnist
    # train_set_X = 2* np.round(train_set_X) -1
    # valid_set_X = 2* np.round(valid_set_X) -1
    # test_set_X = 2* np.round(test_set_X) -1
    # print(np.mean(np.abs(train_set_X)))
    
    # flatten targets
    train_set_t = np.hstack(train_set[1])
    valid_set_t = np.hstack(valid_set[1])
    test_set_t = np.hstack(test_set[1])
    
    # Onehot the targets
    train_set_t = np.float32(np.eye(10)[train_set_t])    
    valid_set_t = np.float32(np.eye(10)[valid_set_t])
    test_set_t = np.float32(np.eye(10)[test_set_t])
    
    print('Building the MLP...') 
    
    # Prepare Theano variables for inputs and targets
    X = T.matrix('X')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)
    
    # input layer
    l = lasagne.layers.InputLayer(shape=(None, 784),input_var=X)
    
    # hidden layer
    l = custom_layers.QuantizedDenseLayer(l, nonlinearity=lasagne.nonlinearities.identity, num_units=num_units, b = None)
    # l = lasagne.layers.BatchNormLayer(l)
    l = lasagne.layers.NonlinearityLayer(l,nonlinearity=T.nnet.relu)
    
    # hidden layer
    l = custom_layers.QuantizedDenseLayer(l, nonlinearity=lasagne.nonlinearities.identity, num_units=num_units, b = None)
    # l = lasagne.layers.BatchNormLayer(l)    
    l = lasagne.layers.NonlinearityLayer(l,nonlinearity=T.nnet.relu)

    # output layer
    l = custom_layers.QuantizedDenseLayer(l, nonlinearity=lasagne.nonlinearities.identity,num_units=10, b = None)
    # l = lasagne.layers.BatchNormLayer(l)
    l = lasagne.layers.NonlinearityLayer(l,nonlinearity=lasagne.nonlinearities.sigmoid)
  
    def loss(t,y):
      return T.mean(T.nnet.binary_crossentropy(y, t))
    
    train_output = lasagne.layers.get_output(l, inputs = X, deterministic=False)
    train_loss = loss(target,train_output)
    
    # adversarial objective
    # as in http://arxiv.org/pdf/1510.04189.pdf
    adversarial_X = theano.gradient.disconnected_grad(X + 0.08 * T.sgn(theano.gradient.grad(cost=train_loss,wrt=X)))
    train_output = lasagne.layers.get_output(l, inputs = adversarial_X, deterministic=False)
    train_loss = loss(target,train_output)
    
    # Parameters updates
    params = lasagne.layers.get_all_params(l,trainable=True)
    updates = lasagne.updates.adam(loss_or_grads=train_loss, params=params, learning_rate=LR)
    # updates = custom_layers.clipping_scaling(updates,l)
    
    # error rate
    test_output = lasagne.layers.get_output(l, deterministic=True)
    test_loss = loss(target,test_output)
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([X, target, LR], test_loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([X, target], [test_loss, test_err])

    print('Training...')
    
    custom_layers.train(
            train_fn,val_fn,
            l,
            batch_size,
            LR_start,LR_decay,LR_patience,
            patience,
            train_set_X,train_set_t,
            valid_set_X,valid_set_t,
            test_set_X,test_set_t,
            save_path,
            shuffle_parts)
            