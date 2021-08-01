#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras.layers import Activation, Add, Dense, BatchNormalization, Concatenate, Dropout, Subtract, Flatten, Input, Lambda, Reshape
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, MaxPool3D, AveragePooling3D, UpSampling3D, ConvLSTM2D
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, AveragePooling2D, UpSampling2D
from tensorflow.keras.layers import Layer, RepeatVector, Permute, Multiply, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint, Callback, CSVLogger
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras import backend as K

import scipy.io as sio
import tensorflow as tf
import numpy as np
import sys 


class BiasLayer(Layer):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.reg = 1e-4

    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=True,
                                    regularizer=regularizers.l2(self.reg))
        
    def call(self, x):
        return x + self.bias


class SeparableConv3D(Layer):
    def __init__(self, conv_filter, kernel_size, strides, dilation_rate, use_bias, is_separable, reg=1e-4):
        super(SeparableConv3D, self).__init__()
        
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_bias = use_bias
        self.conv_filter = conv_filter
        self.is_separable = is_separable
        
        if self.is_separable == True:
            self.Wxy = Conv3D(conv_filter, (1, self.kernel_size, self.kernel_size), 
                              strides=self.strides, 
                              dilation_rate = self.dilation_rate,
                              padding='same', 
                              kernel_regularizer=regularizers.l2(reg), 
                              use_bias=self.use_bias, 
                              bias_regularizer=regularizers.l2(reg)) 
            self.Wz = Conv3D(conv_filter, (4,1,1), 
                             strides=self.strides, 
                             dilation_rate = self.dilation_rate,
                             padding='same', 
                             kernel_regularizer=regularizers.l2(reg), 
                             use_bias=self.use_bias, 
                             bias_regularizer=regularizers.l2(reg))
        
        elif self.is_separable == False:
            self.Wxyz = Conv3D(conv_filter, (self.kernel_size, self.kernel_size, self.kernel_size), 
                               strides=self.strides, 
                               dilation_rate = self.dilation_rate,
                               padding='same', 
                               kernel_regularizer=regularizers.l2(reg), 
                               use_bias=self.use_bias, 
                               bias_regularizer=regularizers.l2(reg))
            
    def call(self, p):
        if self.is_separable == True:
            qxy = self.Wxy(p)
            qz = self.Wz(p)
            
            out = Add()([qxy, qz])
            
        elif self.is_separable == False:
            out = self.Wxyz(p)
            
        return out


class SeparableConv3DTranspose(Layer):
    def __init__(self, conv_filter, kernel_size, strides, dilation_rate, use_bias, is_separable, reg=1e-4):
        super(SeparableConv3DTranspose, self).__init__()
        
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_bias = use_bias
        self.conv_filter = conv_filter
        self.is_separable = is_separable
        
        if self.is_separable == True:
            self.Wxy = Conv3DTranspose(conv_filter, (1, self.kernel_size, self.kernel_size), 
                                       strides=self.strides, 
                                       dilation_rate = self.dilation_rate,
                                       padding='same', 
                                       kernel_regularizer=regularizers.l2(reg), 
                                       use_bias=self.use_bias, 
                                       bias_regularizer=regularizers.l2(reg)) 
            self.Wz = Conv3DTranspose(conv_filter, (4,1,1), strides=self.strides, 
                                      dilation_rate = self.dilation_rate,
                                      padding='same', 
                                      kernel_regularizer=regularizers.l2(reg), 
                                      use_bias=self.use_bias, 
                                      bias_regularizer=regularizers.l2(reg))
        
        elif self.is_separable == False:
            self.Wxyz = Conv3DTranspose(conv_filter, (self.kernel_size, self.kernel_size, self.kernel_size), 
                                        strides=self.strides, 
                                        dilation_rate = self.dilation_rate,
                                        padding='same', 
                                        kernel_regularizer=regularizers.l2(reg), 
                                        use_bias=self.use_bias, 
                                        bias_regularizer=regularizers.l2(reg))
            
    def call(self, p):
        if self.is_separable == True:
            qxy = self.Wxy(p)
            qz = self.Wz(p)
            
            out = Add()([qxy, qz])
            
        elif self.is_separable == False:
            out = self.Wxyz(p)
            
        return out

