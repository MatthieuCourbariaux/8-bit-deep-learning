import time

from collections import OrderedDict

import numpy as np

import theano
import theano.tensor as T
from theano.tensor import as_tensor_variable
from theano.gradient import disconnected_type
import theano.sandbox.cuda as cuda
from theano.sandbox.cuda.basic_ops import host_from_gpu
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
srng=RandomStreams(123456,use_cuda = True)

def stochastic_rounding(x):
  xr = T.floor(x)
  xr += T.cast(srng.binomial(n=1, p=(x - xr), size=T.shape(xr)), theano.config.floatX)
  return xr
  
def quantizer(x,N=8,new_min=-1.,new_max=1.,stochastic=False):
  
  # [min,max] -> [new_min,new_max]
  xq = T.clip(x,new_min,new_max)
  # [new_min,new_max] -> [0,1]
  xq = (xq - new_min) / (new_max - new_min)
  
  # [0,1] -> [0,(2.**N)-1.]
  xq = ((2.**N)-1.) * xq
  
  # rounding
  if stochastic:
    xq = stochastic_rounding(xq)
  else:
    xq = T.round(xq)
  
  # cast to 8-bit
  xq = T.cast(xq,'uint8')
  
  return xq
  
def quantized_dot(x, y, x_N=8, x_min=-1., x_max=1., y_N=8, y_min=-1.,y_max=1.):
    
    # x and y are unint8 variables
    # TODO: write a custom 8-bit dot product
    z = T.dot(T.cast(x,'float32'),T.cast(y,'float32'))
    
    # element-wise stuff
    x_scale = (x_max-x_min)/((2.**x_N)-1.)
    y_scale = (y_max-y_min)/((2.**y_N)-1.)
    z = x_scale*y_scale*z
    z += x_scale * y_min * T.shape_padright(T.sum(T.cast(x,'float32'),axis=1))
    z += y_scale * x_min * T.sum(T.cast(y,'float32'),axis=0)
    z += T.cast(T.shape(x)[1],'float32') * y_min*x_min
    
    return z

class QuantizedGemm(cuda.GpuOp):
    
  def __eq__(self, other):
    return type(self) == type(other)

  def __hash__(self):
    return hash(type(self))

  def __str__(self):
    return self.__class__.__name__
  
  def output_type(self, inp):
    return cuda.CudaNdarrayType(broadcastable=[False, False])
  
  def make_node(self, x, w):
    
    # quantizing the inputs
    xmax = 8.*T.std(x)
    xmin = T.constant(0.) # we are assuming the input is positive
    xq = quantizer(x,new_min=xmin,new_max=xmax)
    
    # quantizing the weights
    wmax = 8.*T.std(w)
    wmin = -wmax
    wq = quantizer(w,new_min=wmin,new_max=wmax)
    
    # low precision dot product
    z = quantized_dot(xq, wq, x_min=xmin, x_max=xmax, y_min=wmin, y_max=wmax)
    z = cuda.basic_ops.gpu_contiguous(cuda.basic_ops.as_cuda_ndarray_variable(z))
    
    return theano.Apply(self, [x,w,xq, xmin, xmax, wq, wmin, wmax,z], [self.output_type(z)()])
  
  # Here make_thunk only passes the 5tht input (the dot product result) as output
  def make_thunk(self, node, storage_map, _, _2):

    inputs = [storage_map[v] for v in node.inputs]
    outputs = [storage_map[v] for v in node.outputs]
     
    def thunk():
      outputs[0][0] = inputs[8][0]

    return thunk
  
  def grad(self, inp, grads):
  
    x, w, xq, xmin, xmax, wq, wmin, wmax, z = inp
    gz, = grads
  
    # quantizing Output's gradient
    gzmax = 8.* T.std(gz)
    gzmin = -gzmax
    # gzq = gz
    gzq = quantizer(gz,new_min=gzmin,new_max=gzmax, stochastic=True)
    
    # Inputs' gradient
    # Low precision dot product
    gxq = quantized_dot(gzq, wq.T, x_min = gzmin, x_max=gzmax, y_min=wmin, y_max=wmax)
    gx = gxq
    
    # Weights' gradient
    # Low precision dot product
    gwq = quantized_dot(xq.T, gzq, x_min = xmin, x_max=xmax, y_min=gzmin, y_max=gzmax)
    gw = gwq 
    
    return gx, gw, gxq, disconnected_type(), disconnected_type(), gwq, disconnected_type(), disconnected_type(), gz
    
quantized_gemm = QuantizedGemm()
