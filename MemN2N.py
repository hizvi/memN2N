import numpy as np

from keras.models import Model
from keras.layers import Dense, Embedding, Input, Dropout, Lambda, Flatten
from keras.layers import Dot, Activation, Softmax, Add, Multiply, Permute
from keras.layers import dot, add, multiply

from keras.engine.topology import Layer

from keras.backend import variable, transpose, reshape, gather
from keras import backend as K


class MemN2NBlock(Layer): 
    def __init__(self, output_dim):
        super(MemN2NBlock, self).__init__()
        
        self.output_dim = output_dim
        
        # layer operations
        self.input_memory = Dot(axes=(-1))
        self.input_representation = Softmax()
        self.permute_weights = Permute((2,1))
        self.output_memory = Dot(axes=(2,1))
        self.h_mapping = self.add_weight(
                name='H',
                shape=(self.output_dim[2],output_dim[2]),
                initializer='glorot_normal',
                trainable=True
        )
        self.new_u = Add()
        
    def call(self, inputs):
        m = self.input_memory([inputs[0], inputs[1]])
        p = self.input_representation(m)
        # print('p.shape: ', p.shape)
        # p = self.permute_weights(p) 
        p = reshape(p, [-1, p.shape[2], p.shape[1]])
        # print('p.shape: ', p.shape)
        c = self.output_memory([p, inputs[2]])
        
        mapped_u = K.dot(inputs[1], self.h_mapping)
        
        return self.new_u([c, mapped_u])
        
    def build(self, input_shape):
        super(MemN2NBlock, self).build(input_shape)
        
    def compute_output_shape(self, input_shape):
        return self.output_dim
 