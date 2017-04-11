import numpy as np
# from keras.layers.core import  Lambda, Merge
from keras.layers import Conv2D
from keras import backend as K

from keras.engine import Layer

class Softmax4D(Layer):
    def __init__(self, axis=-1,**kwargs):
        self.axis=axis
        super(Softmax4D, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, x,mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s

    def compute_output_shape(self, input_shape):
        return input_shape

class gramMatrix(Layer):
    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
#        print(shape)
        assert len(shape) == 4  # only valid for 2D tensors
        shape = (shape[0], shape[3], shape[3], 1)
        return shape

    def call(self, x, mask=None):
        x = K.permute_dimensions(x, (0, 3, 1, 2))
        my_shape = K.int_shape(x)
        features = K.reshape(x, shape=(-1,my_shape[1], my_shape[2]*my_shape[3]))
        gram = K.batch_dot(features, K.permute_dimensions(features, (0,2,1)))
        gram = K.expand_dims(gram)
        return gram

class gramSpatial(Layer):
    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
#        print(shape)
        assert len(shape) == 4  # only valid for 2D tensors
        shape = (shape[0], shape[1]*shape[2], shape[1]*shape[2], 1)
        return shape

    def call(self, x, mask=None):
        # x = K.permute_dimensions(x, (0, 3, 1, 2))
        my_shape = K.int_shape(x)
        features = K.reshape(x, shape=(-1,my_shape[1]*my_shape[2], my_shape[3]))
        gram = K.batch_dot(features, K.permute_dimensions(features, (0,2,1)))
        gram = K.expand_dims(gram)
        return gram    