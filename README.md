# 8-bit Deep Learning

## Motivations

In this repository, we train an MLP with 8-bit multiplications (but 32-bit accumulations).
Our method could take advantage of a Pascal (the latest Nvidia GPU architecture) instruction
which performs 4 x 8-bit multiplications 32-bit accumulation per clock cycle.
Basically, we could train neural networks about 4 times faster than float32 with no loss in accuracy.

## Requirements

* Python 2.7, Numpy, Scipy
* [Theano](http://deeplearning.net/software/theano/install.html)
* A fast Nvidia GPU (or a large amount of patience)
* Setting your [Theano flags](http://deeplearning.net/software/theano/library/config.html) to use the GPU
* [Lasagne](http://lasagne.readthedocs.org/en/latest/user/installation.html)

## MNIST MLP

Firstly, download the MNIST dataset:
    
    wget http://deeplearning.net/data/mnist/mnist.pkl.gz
    
Then, simply run the training script:

    python mnist_mlp.py
    
It should run for about 15 minutes on a Titan X GPU.
The final test error should be around **0.96%**.

## CIFAR-10 ConvNet

TODO

## ImageNet ConvNet

TODO