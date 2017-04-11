# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 22:52:34 2016

@author: eunsoo
"""
#%%
from keras.layers import Input, merge, SeparableConv2D, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dropout, Activation
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model

#%%
sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"
batchN = "batch_"
#%%
def fireXcep_module(x, fire_id, squeeze=16, expand=64, dim_ordering='tf'):
    s_id = 'fire'+str(fire_id)+'/'
    if dim_ordering is 'tf':
        c_axis = 3
    else:
        c_axis = 1
    
    x = Conv2D(squeeze, 1, 1, padding='valid', name=s_id+sq1x1)(x)
    x = BatchNormalization(name=s_id+batchN+sq1x1)(x)
    x = Activation('relu', name=s_id+relu+sq1x1)(x)
    
    left = Conv2D(expand, 1, 1, padding='valid', name=s_id+exp1x1)(x)
    left = BatchNormalization(name=s_id+batchN+exp1x1)(left)
    left = Activation('relu', name=s_id+relu+exp1x1)(left)
    
    right = SeparableConv2D(expand, 3, 3, padding='same', name=s_id+exp3x3)(x)
    right = BatchNormalization(name=s_id+batchN+exp3x3)(right)
    right = Activation('relu', name=s_id+relu+exp3x3)(right)
    
    x = merge([left, right], mode='concat', concat_axis=c_axis, name=s_id+'concat')
    return x


def fireXcep_module2(x, fire_id, squeeze=16, expand=64, dim_ordering='tf'):
    s_id = 'fire'+str(fire_id)+'/'
    if dim_ordering is 'tf':
        c_axis = 3
    else:
        c_axis = 1
    
    x = Conv2D(squeeze, 1, 1, padding='valid', name=s_id+sq1x1)(x)
    x = BatchNormalization(name=s_id+batchN+sq1x1)(x)
    x = Activation('relu', name=s_id+relu+sq1x1)(x)
    
    left = Conv2D(expand, 1, 1, padding='valid', name=s_id+exp1x1)(x)
    left = BatchNormalization(name=s_id+batchN+exp1x1)(left)
    left = Activation('relu', name=s_id+relu+exp1x1)(left)
    
    right = SeparableConv2D(expand*3, 3, 3, padding='same', name=s_id+exp3x3)
    right = BatchNormalization(name=s_id+batchN+exp3x3)(right)
    right = Activation('relu', name=s_id+relu+exp3x3)(right)
    
    x = merge([left, right], mode='concat', concat_axis=c_axis, name=s_id+'concat')
    return x

# Modular function for Fire Node
def fire_module(x, fire_id, squeeze=16, expand=64, dim_ordering='tf'):
    s_id = 'fire'+str(fire_id)+'/'
    if dim_ordering is 'tf':
        c_axis = 3
    else:
        c_axis = 1
    
    x = Conv2D(squeeze, 1, 1, padding='valid', name=s_id+sq1x1)(x)
    x = BatchNormalization(name=s_id+batchN+sq1x1)(x)
    x = Activation('relu', name=s_id+relu+sq1x1)(x)
    
    left = Conv2D(expand, 1, 1, padding='valid', name=s_id+exp1x1)(x)
    x = BatchNormalization(name=s_id+batchN+exp1x1)(x)
    left = Activation('relu', name=s_id+relu+exp1x1)(left)
    
    right = Conv2D(expand, 3, 3, padding='same', name=s_id+exp3x3)(x)
    x = BatchNormalization(name=s_id+batchN+exp3x3)(x)
    right = Activation('relu', name=s_id+relu+exp3x3)(right)
    
    x = merge([left, right], mode='concat', concat_axis=c_axis, name=s_id+'concat')
    return x
#%%    
# Original Squeeze from paper. Updated version from squeezenet paper.
def get_squeezenet(nb_classes, dim_ordering='tf'):
    if dim_ordering is 'th':
        input_img = Input(shape=(3, 224, 224))
    elif dim_ordering is 'tf':
        input_img = Input(shape=(224, 224, 3))
    else:
        raise NotImplementedError("Theano and Tensorflow are only avaiable")
    x = Conv2D(96, 7, 7, strides=(2,2), padding='same',name='conv1_park')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)
    # x = Conv2D(64, 3, 3, strides=(2, 2), padding='valid', name='conv1')(input_img)    
    x = Activation('relu', name='relu_conv1')(x)    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
    
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    x = Dropout(0.5, name='drop9')(x)    
    x = Conv2D(nb_classes, 1, 1, padding='valid', name='conv10_park')(x)
    x = BatchNormalization(name='batch_conv10')(x)

    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)
    
    model = Model(input=input_img, output=[out])

    return model

# Original Squeeze from paper. Updated version from squeezenet paper.
def get_squeezeXceNet(nb_classes, dim_ordering='tf'):
    if dim_ordering is 'th':
        input_img = Input(shape=(3, 224, 224))
    elif dim_ordering is 'tf':
        input_img = Input(shape=(224, 224, 3))
    else:
        raise NotImplementedError("Theano and Tensorflow are only avaiable")
    x = Conv2D(96, 7, 7, strides=(2,2), padding='same', name='conv1_park')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)    
    x = Activation('relu', name='relu_conv1')(x)    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
    x = fireXcep_module(x, fire_id=2, squeeze=16, expand=64)
    x = fireXcep_module(x, fire_id=3, squeeze=16, expand=64)
    x = fireXcep_module(x, fire_id=4, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    
    x = fireXcep_module(x, fire_id=5, squeeze=32, expand=128)
    x = fireXcep_module(x, fire_id=6, squeeze=48, expand=192)
    x = fireXcep_module(x, fire_id=7, squeeze=48, expand=192)
    x = fireXcep_module(x, fire_id=8, squeeze=64, expand=256)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
    
    x = fireXcep_module(x, fire_id=9, squeeze=64, expand=256)
    x = Dropout(0.5, name='drop9')(x)    
    x = Conv2D(nb_classes, 1, 1, padding='valid', name='conv10_park')(x)
    x = BatchNormalization(name='batch_conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)
    
    model = Model(input=input_img, output=[out])

    return model

def get_sqWideSr50(nb_classes, dim_ordering='tf'):
    if dim_ordering is 'th':
        input_img = Input(shape=(3, 224, 224))
    elif dim_ordering is 'tf':
        input_img = Input(shape=(224, 224, 3))
    else:
        raise NotImplementedError("Theano and Tensorflow are only avaiable")
    x = SeparableConv2D(96, 7, 7, strides=(2,2), padding='same', name='conv1_park')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)    
    x = Activation('relu', name='relu_conv1')(x)    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
    x = fireXcep_module(x, fire_id=2, squeeze=32, expand=64)
    x = fireXcep_module(x, fire_id=3, squeeze=32, expand=64)
    x = fireXcep_module(x, fire_id=4, squeeze=64, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    
    x = fireXcep_module(x, fire_id=5, squeeze=64, expand=128)
    x = fireXcep_module(x, fire_id=6, squeeze=128, expand=256)
    x = fireXcep_module(x, fire_id=7, squeeze=128, expand=256)
    x = fireXcep_module(x, fire_id=8, squeeze=256, expand=512)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
    
    x = fireXcep_module(x, fire_id=9, squeeze=256, expand=512)
    x = Dropout(0.5, name='drop9')(x)    
    x = Conv2D(nb_classes, 1, 1, padding='valid', name='conv10_park')(x)
    x = BatchNormalization(name='batch_conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)
    
    model = Model(input=input_img, output=[out])

    return model


def get_sqAllXceNet(nb_classes, dim_ordering='tf'):
    if dim_ordering is 'th':
        input_img = Input(shape=(3, 224, 224))
    elif dim_ordering is 'tf':
        input_img = Input(shape=(224, 224, 3))
    else:
        raise NotImplementedError("Theano and Tensorflow are only avaiable")
    x = SeparableConv2D(96, 7, 7, strides=(2,2), padding='same', name='conv1_park')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)    
    x = Activation('relu', name='relu_conv1')(x)    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
    x = fireXcep_module(x, fire_id=2, squeeze=16, expand=64)
    x = fireXcep_module(x, fire_id=3, squeeze=16, expand=64)
    x = fireXcep_module(x, fire_id=4, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    
    x = fireXcep_module(x, fire_id=5, squeeze=32, expand=128)
    x = fireXcep_module(x, fire_id=6, squeeze=48, expand=192)
    x = fireXcep_module(x, fire_id=7, squeeze=48, expand=192)
    x = fireXcep_module(x, fire_id=8, squeeze=64, expand=256)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
    
    x = fireXcep_module(x, fire_id=9, squeeze=64, expand=256)
    x = Dropout(0.5, name='drop9')(x)    
    x = Conv2D(nb_classes, 1, 1, padding='valid', name='conv10_park')(x)
    x = BatchNormalization(name='batch_conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)
    
    model = Model(input=input_img, output=[out])

    return model

def get_sqAllXceNet_depth(nb_classes, dim_ordering='tf'):
    if dim_ordering is 'th':
        input_img = Input(shape=(3, 224, 224))
    elif dim_ordering is 'tf':
        input_img = Input(shape=(224, 224, 3))
    else:
        raise NotImplementedError("Theano and Tensorflow are only avaiable")
    x = SeparableConv2D(96, 7, 7, strides=(2,2), padding='same', name='conv1_park')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)    
    x = Activation('relu', name='relu_conv1')(x)    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
    x = fireXcep_module2(x, fire_id=2, squeeze=16, expand=64)
    x = fireXcep_module2(x, fire_id=3, squeeze=16, expand=64)
    x = fireXcep_module2(x, fire_id=4, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    
    x = fireXcep_module2(x, fire_id=5, squeeze=32, expand=128)
    x = fireXcep_module2(x, fire_id=6, squeeze=48, expand=192)
    x = fireXcep_module2(x, fire_id=7, squeeze=48, expand=192)
    x = fireXcep_module2(x, fire_id=8, squeeze=64, expand=256)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
    
    x = fireXcep_module2(x, fire_id=9, squeeze=64, expand=256)
    x = Dropout(0.5, name='drop9')(x)    
    x = Conv2D(nb_classes, 1, 1, padding='valid', name='conv10_park')(x)
    x = BatchNormalization(name='batch_conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)
    
    model = Model(input=input_img, output=[out])

    return model


def get_sqAllXceNet_2(nb_classes, dim_ordering='tf'):
    if dim_ordering is 'th':
        input_img = Input(shape=(3, 224, 224))
    elif dim_ordering is 'tf':
        input_img = Input(shape=(224, 224, 3))
    else:
        raise NotImplementedError("Theano and Tensorflow are only avaiable")
    x = SeparableConv2D(96, 7, 7, strides=(2,2), padding='same', name='conv1_park')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)    
    x = Activation('relu', name='relu_conv1')(x)    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
    x = fireXcep_module(x, fire_id=2, squeeze=16, expand=64)
    x = fireXcep_module(x, fire_id=3, squeeze=16, expand=64)
    x = fireXcep_module(x, fire_id=4, squeeze=16, expand=64)
    x = fireXcep_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    
    x = fireXcep_module(x, fire_id=6, squeeze=32, expand=128)
    x = fireXcep_module(x, fire_id=7, squeeze=32, expand=128)
    x = fireXcep_module(x, fire_id=8, squeeze=48, expand=192)
    x = fireXcep_module(x, fire_id=9, squeeze=48, expand=192)
    x = fireXcep_module(x, fire_id=10, squeeze=48, expand=192)
    x = fireXcep_module(x, fire_id=11, squeeze=64, expand=256)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
    
    x = fireXcep_module(x, fire_id=12, squeeze=64, expand=256)
    x = fireXcep_module(x, fire_id=13, squeeze=64, expand=256)
    x = Dropout(0.5, name='drop9')(x)    
    x = Conv2D(nb_classes, 1, 1, padding='valid', name='conv10_park')(x)
    x = BatchNormalization(name='batch_conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)
    
    model = Model(input=input_img, output=[out])

    return model

def get_sqAllXceNetFinalDeep(nb_classes, dim_ordering='tf'):
    if dim_ordering is 'th':
        input_img = Input(shape=(3, 224, 224))
    elif dim_ordering is 'tf':
        input_img = Input(shape=(224, 224, 3))
    else:
        raise NotImplementedError("Theano and Tensorflow are only avaiable")
    x = SeparableConv2D(96, 7, 7, strides=(2,2), padding='same', name='conv1_park')(input_img)
    # x = Conv2D(64, 3, 3, strides=(2, 2), padding='valid', name='conv1')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)    
    x = Activation('relu', name='relu_conv1')(x)    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
    x = fireXcep_module(x, fire_id=2, squeeze=16, expand=64)
    residual = x
    x = fireXcep_module(x, fire_id=3, squeeze=16, expand=64)
    x = merge([x, residual], mode='sum')
    residual = x
    x = fireXcep_module(x, fire_id=4, squeeze=16, expand=64)
    x = merge([x, residual], mode='sum')
    residual = x
    x = fireXcep_module(x, fire_id=5, squeeze=16, expand=64)
    x = merge([x, residual], mode='sum')
    residual = x
    x = fireXcep_module(x, fire_id=6, squeeze=16, expand=64)
    x = merge([x, residual], mode='sum')
    residual = x
    x = fireXcep_module(x, fire_id=7, squeeze=16, expand=64)
    x = merge([x, residual], mode='sum')
    
    x = fireXcep_module(x, fire_id=8, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
    residual = x
    x = fireXcep_module(x, fire_id=9, squeeze=32, expand=128)
    x = merge([x, residual], mode='sum')
    residual = x
    x = fireXcep_module(x, fire_id=10, squeeze=32, expand=128)
    x = merge([x, residual], mode='sum')
    residual = x
    x = fireXcep_module(x, fire_id=11, squeeze=32, expand=128)
    x = merge([x, residual], mode='sum')
    residual = x
    x = fireXcep_module(x, fire_id=12, squeeze=32, expand=128)
    x = merge([x, residual], mode='sum')
    residual = x
    x = fireXcep_module(x, fire_id=13, squeeze=32, expand=128)
    x = merge([x, residual], mode='sum')
    
    x = fireXcep_module(x, fire_id=14, squeeze=48, expand=192)
    residual = x
    x = fireXcep_module(x, fire_id=15, squeeze=48, expand=192)
    x = merge([x, residual], mode='sum')
    residual = x
    x = fireXcep_module(x, fire_id=16, squeeze=48, expand=192)
    x = merge([x, residual], mode='sum')
    residual = x
    x = fireXcep_module(x, fire_id=17, squeeze=48, expand=192)
    x = merge([x, residual], mode='sum')
    residual = x
    x = fireXcep_module(x, fire_id=18, squeeze=48, expand=192)
    x = merge([x, residual], mode='sum')
#    residual = x
#    x = fireXcep_module(x, fire_id=19, squeeze=48, expand=192)
#    x = merge([x, residual], mode='sum')
    
    x = fireXcep_module(x, fire_id=20, squeeze=64, expand=256)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool20')(x) 
    residual = x
    x = fireXcep_module(x, fire_id=21, squeeze=64, expand=256)
    x = merge([x, residual], mode='sum')
    residual = x
    x = fireXcep_module(x, fire_id=22, squeeze=64, expand=256)
    x = merge([x, residual], mode='sum')
    residual = x
    x = fireXcep_module(x, fire_id=23, squeeze=64, expand=256)
    x = merge([x, residual], mode='sum')
    residual = x
    x = fireXcep_module(x, fire_id=24, squeeze=64, expand=256)
    x = merge([x, residual], mode='sum')
    residual = x
    x = fireXcep_module(x, fire_id=25, squeeze=64, expand=256)
    x = merge([x, residual], mode='sum')

    x = Dropout(0.5, name='drop25')(x)    
    x = Conv2D(nb_classes, 1, 1, padding='valid', name='conv18_park')(x)
    x = BatchNormalization(name='batch_conv18')(x)
    x = Activation('relu', name='relu_conv18')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)
    
    model = Model(input=input_img, output=[out])

    return model  

def get_sqAllXceNetFinalWide(nb_classes, dim_ordering='tf'):
    mul = 96
    if dim_ordering is 'th':
        input_img = Input(shape=(3, 224, 224))
    elif dim_ordering is 'tf':
        input_img = Input(shape=(224, 224, 3))
    else:
        raise NotImplementedError("Theano and Tensorflow are only avaiable")
    x = SeparableConv2D(96+mul, 7, 7, strides=(2,2), padding='same', name='conv1_park')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)    
    x = Activation('relu', name='relu_conv1')(x)    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
    x = fireXcep_module(x, fire_id=2, squeeze=32, expand=64+mul)
    x = fireXcep_module(x, fire_id=3, squeeze=32, expand=64+mul)
    x = fireXcep_module(x, fire_id=4, squeeze=64, expand=128+mul)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    
    x = fireXcep_module(x, fire_id=5, squeeze=64, expand=128+mul)
    x = fireXcep_module(x, fire_id=6, squeeze=96, expand=192+mul)
    x = fireXcep_module(x, fire_id=7, squeeze=96, expand=192+mul)
    x = fireXcep_module(x, fire_id=8, squeeze=128, expand=256+mul)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
    
    x = fireXcep_module(x, fire_id=9, squeeze=128, expand=256+mul)
    x = Dropout(0.5, name='drop9')(x)    
    x = Conv2D(nb_classes, 1, 1, padding='valid', name='conv10_park')(x)
    x = BatchNormalization(name='batch_conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)
    
    model = Model(input=input_img, output=[out])

    return model

def get_sqAllXceNetSr50(nb_classes, dim_ordering='tf'):
    if dim_ordering is 'th':
        input_img = Input(shape=(3, 224, 224))
    elif dim_ordering is 'tf':
        input_img = Input(shape=(224, 224, 3))
    else:
        raise NotImplementedError("Theano and Tensorflow are only avaiable")
    x = SeparableConv2D(96, 7, 7, strides=(2,2), padding='same', name='conv1_park')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)    
    x = Activation('relu', name='relu_conv1')(x)    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
    x = fireXcep_module(x, fire_id=2, squeeze=32, expand=64)
    x = fireXcep_module(x, fire_id=3, squeeze=32, expand=64)
    x = fireXcep_module(x, fire_id=4, squeeze=64, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    
    x = fireXcep_module(x, fire_id=5, squeeze=64, expand=128)
    x = fireXcep_module(x, fire_id=6, squeeze=96, expand=192)
    x = fireXcep_module(x, fire_id=7, squeeze=96, expand=192)
    x = fireXcep_module(x, fire_id=8, squeeze=128, expand=256)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
    
    x = fireXcep_module(x, fire_id=9, squeeze=128, expand=256)
    x = Dropout(0.5, name='drop9')(x)    
    x = Conv2D(nb_classes, 1, 1, padding='valid', name='conv10_park')(x)
    x = BatchNormalization(name='batch_conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)
    
    model = Model(input=input_img, output=[out])

    return model

# Original Squeeze from paper. Updated version from squeezenet paper.
def get_sqXceNetRes(nb_classes, dim_ordering='tf'):
    if dim_ordering is 'th':
        input_img = Input(shape=(3, 224, 224))
    elif dim_ordering is 'tf':
        input_img = Input(shape=(224, 224, 3))
    else:
        raise NotImplementedError("Theano and Tensorflow are only avaiable")
    x = Conv2D(96, 7, 7, strides=(2,2), padding='same', name='conv1_park')(input_img)
    # x = Conv2D(64, 3, 3, strides=(2, 2), padding='valid', name='conv1')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)    
    x = Activation('relu', name='relu_conv1')(x)    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
    x = fireXcep_module(x, fire_id=2, squeeze=16, expand=64)
    residual = x
    x = fireXcep_module(x, fire_id=3, squeeze=16, expand=64)
    x = merge([x, residual], mode='sum')
    x = fireXcep_module(x, fire_id=4, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    
    residual = x
    x = fireXcep_module(x, fire_id=5, squeeze=32, expand=128)
    x = merge([x, residual], mode='sum')
    x = fireXcep_module(x, fire_id=6, squeeze=48, expand=192)

    residual = x
    x = fireXcep_module(x, fire_id=7, squeeze=48, expand=192)
    x = merge([x, residual], mode='sum')

    x = fireXcep_module(x, fire_id=8, squeeze=64, expand=256)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
    
    residual = x
    x = fireXcep_module(x, fire_id=9, squeeze=64, expand=256)
    x = merge([x, residual], mode='sum')
    
    x = Dropout(0.5, name='drop9')(x)    
    x = Conv2D(nb_classes, 1, 1, padding='valid', name='conv10_park')(x)
    x = BatchNormalization(name='batch_conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)
    
    model = Model(input=input_img, output=[out])

    return model  

def get_sqAllXResNet(nb_classes, dim_ordering='tf'):
    if dim_ordering is 'th':
        input_img = Input(shape=(3, 224, 224))
    elif dim_ordering is 'tf':
        input_img = Input(shape=(224, 224, 3))
    else:
        raise NotImplementedError("Theano and Tensorflow are only avaiable")
    x = SeparableConv2D(96, 7, 7, strides=(2,2), padding='same', name='conv1_park')(input_img)
    # x = Conv2D(64, 3, 3, strides=(2, 2), padding='valid', name='conv1')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)    
    x = Activation('relu', name='relu_conv1')(x)    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
    x = fireXcep_module(x, fire_id=2, squeeze=16, expand=64)
    residual = x
    x = fireXcep_module(x, fire_id=3, squeeze=16, expand=64)
    x = merge([x, residual], mode='sum')
    x = fireXcep_module(x, fire_id=4, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    
    residual = x
    x = fireXcep_module(x, fire_id=5, squeeze=32, expand=128)
    x = merge([x, residual], mode='sum')
    x = fireXcep_module(x, fire_id=6, squeeze=48, expand=192)

    residual = x
    x = fireXcep_module(x, fire_id=7, squeeze=48, expand=192)
    x = merge([x, residual], mode='sum')

    x = fireXcep_module(x, fire_id=8, squeeze=64, expand=256)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
    
    residual = x
    x = fireXcep_module(x, fire_id=9, squeeze=64, expand=256)
    x = merge([x, residual], mode='sum')
    
    x = Dropout(0.5, name='drop9')(x)    
    x = Conv2D(nb_classes, 1, 1, padding='valid', name='conv10_park')(x)
    x = BatchNormalization(name='batch_conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)
    
    model = Model(input=input_img, output=[out])

    return model  

def get_sqAllXResNet_2(nb_classes, dim_ordering='tf'):
    if dim_ordering is 'th':
        input_img = Input(shape=(3, 224, 224))
    elif dim_ordering is 'tf':
        input_img = Input(shape=(224, 224, 3))
    else:
        raise NotImplementedError("Theano and Tensorflow are only avaiable")
    x = SeparableConv2D(96, 7, 7, strides=(2,2), padding='same', name='conv1_park')(input_img)
    # x = Conv2D(64, 3, 3, strides=(2, 2), padding='valid', name='conv1')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)    
    x = Activation('relu', name='relu_conv1')(x)    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
    x = fireXcep_module(x, fire_id=2, squeeze=16, expand=64)
    residual = x
    x = fireXcep_module(x, fire_id=3, squeeze=16, expand=64)
    x = merge([x, residual], mode='sum')
    
    residual = x
    x = fireXcep_module(x, fire_id=4, squeeze=16, expand=64)
    x = merge([x, residual], mode='sum')
    
    x = fireXcep_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)
    
    residual = x
    x = fireXcep_module(x, fire_id=6, squeeze=32, expand=128)
    x = merge([x, residual], mode='sum')
    
    residual = x
    x = fireXcep_module(x, fire_id=7, squeeze=32, expand=128)
    x = merge([x, residual], mode='sum')

    x = fireXcep_module(x, fire_id=8, squeeze=48, expand=192)

    residual = x
    x = fireXcep_module(x, fire_id=9, squeeze=48, expand=192)
    x = merge([x, residual], mode='sum')
    
    residual = x
    x = fireXcep_module(x, fire_id=10, squeeze=48, expand=192)
    x = merge([x, residual], mode='sum')

    x = fireXcep_module(x, fire_id=11, squeeze=64, expand=256)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool9')(x)
    
    residual = x
    x = fireXcep_module(x, fire_id=12, squeeze=64, expand=256)
    x = merge([x, residual], mode='sum')

    residual = x
    x = fireXcep_module(x, fire_id=13, squeeze=64, expand=256)
    x = merge([x, residual], mode='sum')

    x = Dropout(0.5, name='drop9')(x)    
    x = Conv2D(nb_classes, 1, 1, padding='valid', name='conv14_park')(x)
    x = BatchNormalization(name='batch_conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)
    
    model = Model(input=input_img, output=[out])

    return model  



#%%    
if __name__ == '__main__':
    import time
#    from keras.utils.visualize_util import plot
    start = time.time()
    model = get_squeezeXceNet(2)
    duration = time.time() - start
    print "{} s to make model".format(duration)
    
    start = time.time()
    model.output
    duration = time.time()- start
    print "{} s to get output.".format(duration)
    
    start = time.time()
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    duration = time.time() - start
    print "{} s to get compile".format(duration)
    
#    plot(model, to_file='images/SqueezeNet_new.png', show_shapes=True)
    
    
    
    
    
    
    