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

from separable_layers import BiasLayer, SeparableConv3D, SeparableConv3DTranspose

import scipy.io as sio
import tensorflow as tf
import numpy as np
import sys 


class SeparableConvGRU3D(Layer):
    def __init__(self, convgru3d_filter, is_separable, reg=1e-4):
        super(SeparableConvGRU3D, self).__init__()
        
        self.is_separable = is_separable
        self.convgru3d_filter = convgru3d_filter
        self.kernel_size = 3
        self.strides = (1,1,1)
        self.dilation_rate = (1,1,1)
        
        self.sigmoid = Activation('sigmoid')
        self.tanh = Activation('tanh')
        self.relu = Activation('relu')
        
        self.Wr = SeparableConv3D(conv_filter=self.convgru3d_filter[0], 
                                  kernel_size=self.kernel_size, 
                                  strides=self.strides, 
                                  dilation_rate=self.dilation_rate,
                                  use_bias=False, 
                                  is_separable=self.is_separable)
        self.Ur = SeparableConv3D(conv_filter=self.convgru3d_filter[0], 
                                  kernel_size=self.kernel_size, 
                                  strides=self.strides, 
                                  dilation_rate=self.dilation_rate,
                                  use_bias=False, 
                                  is_separable=self.is_separable)
        self.br = BiasLayer()
        
        self.Wz = SeparableConv3D(conv_filter=self.convgru3d_filter[0], 
                                  kernel_size=self.kernel_size, 
                                  strides=self.strides, 
                                  dilation_rate=self.dilation_rate,
                                  use_bias=False, 
                                  is_separable=self.is_separable)
        self.Uz = SeparableConv3D(conv_filter=self.convgru3d_filter[0], 
                                  kernel_size=self.kernel_size, 
                                  strides=self.strides, 
                                  dilation_rate=self.dilation_rate,
                                  use_bias=False, 
                                  is_separable=self.is_separable)
        self.bz = BiasLayer()
        
        self.bh = BiasLayer()
        
        self.W = SeparableConv3D(conv_filter=self.convgru3d_filter[0], 
                                 kernel_size=self.kernel_size, 
                                 strides=self.strides, 
                                 dilation_rate=self.dilation_rate,
                                 use_bias=False, 
                                 is_separable=self.is_separable)
        self.U = SeparableConv3D(conv_filter=self.convgru3d_filter[0], kernel_size=self.kernel_size, 
                                 strides=self.strides, 
                                 dilation_rate=self.dilation_rate,
                                 use_bias=False, 
                                 is_separable=self.is_separable)

        
    def call(self, x, h):
        r = self.sigmoid(self.br(Add()([self.Wr(x), self.Ur(h)])))
        z = self.sigmoid(self.bz(Add()([self.Wz(x), self.Uz(h)])))
        r_x_h = self.U(Multiply()([r, h]))
        th = self.relu(self.bh(Add()([self.W(x), r_x_h])))
        
        ones_tensor = tf.constant(value=1.0, shape=z.shape, dtype=z.dtype)
        cz = Subtract()([ones_tensor, z])
        
        z_x_h = Multiply()([z, h])
        cz_x_th = Multiply()([cz, th])
        
        h = Add()([z_x_h, cz_x_th])
        
        return h


# Input sequence: N_view x 64 x 64 x 4 x 1
# Input to ConvLSTM3D: N_view x 4 x 4 x 4 x M
# n_convfilter = [32, 48, 64, 96]
class encoder(Layer):
    def __init__(self, n_convfilter, is_separable, reg=1e-4, dropout_rate=2e-2):
        super(encoder, self).__init__()
        
        self.is_separable = is_separable
        self.n_convfilter = n_convfilter
        
        self.sigmoid = Activation('sigmoid')
        self.tanh = Activation('tanh')
        self.relu = Activation('relu')
        self.drop = Dropout(dropout_rate)
        
        self.bn1a = BatchNormalization()
        self.bn1b = BatchNormalization()
        self.bn1c = BatchNormalization()
        self.bn1d = BatchNormalization()
        self.conv1a = SeparableConv3D(conv_filter=self.n_convfilter[0], kernel_size=3, strides=(1,2,2), dilation_rate = (1,1,1), 
                                      use_bias=True, is_separable=self.is_separable)
        self.conv1b = SeparableConv3D(conv_filter=self.n_convfilter[0], kernel_size=3, strides=(1,1,1), dilation_rate = (1,1,1),
                                      use_bias=True, is_separable=self.is_separable)
        self.conv1c = SeparableConv3D(conv_filter=self.n_convfilter[0], kernel_size=1, strides=(1,2,2), dilation_rate = (1,1,1),
                                      use_bias=True, is_separable=self.is_separable)
        self.conv1d = SeparableConv3D(conv_filter=self.n_convfilter[0], kernel_size=3, strides=(1,1,1), dilation_rate = (1,1,1),
                                      use_bias=True, is_separable=self.is_separable)
        self.conv1e = SeparableConv3D(conv_filter=self.n_convfilter[0], kernel_size=3, strides=(1,1,1), dilation_rate = (1,1,1),
                                      use_bias=True, is_separable=self.is_separable)

        self.bn2a = BatchNormalization()
        self.bn2b = BatchNormalization()
        self.bn2c = BatchNormalization()
        self.bn2d = BatchNormalization()
        self.conv2a = SeparableConv3D(conv_filter=self.n_convfilter[1], kernel_size=3, strides=(1,2,2), dilation_rate = (1,1,1),
                                      use_bias=True, is_separable=self.is_separable)
        self.conv2b = SeparableConv3D(conv_filter=self.n_convfilter[1], kernel_size=3, strides=(1,1,1), dilation_rate = (1,2,2),
                                      use_bias=True, is_separable=self.is_separable)
        self.conv2c = SeparableConv3D(conv_filter=self.n_convfilter[1], kernel_size=1, strides=(1,2,2), dilation_rate = (1,1,1),
                                      use_bias=True, is_separable=self.is_separable)
        self.conv2d = SeparableConv3D(conv_filter=self.n_convfilter[1], kernel_size=3, strides=(1,1,1), dilation_rate = (1,2,2),
                                      use_bias=True, is_separable=self.is_separable)
        self.conv2e = SeparableConv3D(conv_filter=self.n_convfilter[1], kernel_size=3, strides=(1,1,1), dilation_rate = (1,2,2),
                                      use_bias=True, is_separable=self.is_separable)
               
        self.bn3a = BatchNormalization()
        self.bn3b = BatchNormalization()
        self.bn3c = BatchNormalization()
        self.bn3d = BatchNormalization()
        self.conv3a = SeparableConv3D(conv_filter=self.n_convfilter[2], kernel_size=3, strides=(1,2,2), dilation_rate = (1,1,1),
                                      use_bias=True, is_separable=self.is_separable)
        self.conv3b = SeparableConv3D(conv_filter=self.n_convfilter[2], kernel_size=3, strides=(1,1,1), dilation_rate = (1,2,2),
                                      use_bias=True, is_separable=self.is_separable)
        self.conv3c = SeparableConv3D(conv_filter=self.n_convfilter[2], kernel_size=1, strides=(1,2,2), dilation_rate = (1,1,1),
                                      use_bias=True, is_separable=self.is_separable)
        self.conv3d = SeparableConv3D(conv_filter=self.n_convfilter[2], kernel_size=3, strides=(1,1,1), dilation_rate = (1,2,2),
                                      use_bias=True, is_separable=self.is_separable)
        self.conv3e = SeparableConv3D(conv_filter=self.n_convfilter[2], kernel_size=3, strides=(1,1,1), dilation_rate = (1,2,2),
                                      use_bias=True, is_separable=self.is_separable)

        self.bn4a = BatchNormalization()
        self.bn4b = BatchNormalization()
        self.bn4c = BatchNormalization()
        self.bn4d = BatchNormalization()
        self.conv4a = SeparableConv3D(conv_filter=self.n_convfilter[3], kernel_size=3, strides=(1,2,2), dilation_rate = (1,1,1),
                                      use_bias=True, is_separable=self.is_separable)
        self.conv4b = SeparableConv3D(conv_filter=self.n_convfilter[3], kernel_size=3, strides=(1,1,1), dilation_rate = (1,2,2),
                                      use_bias=True, is_separable=self.is_separable)
        self.conv4c = SeparableConv3D(conv_filter=self.n_convfilter[3], kernel_size=1, strides=(1,2,2), dilation_rate = (1,1,1),
                                      use_bias=True, is_separable=self.is_separable)
        self.conv4d = SeparableConv3D(conv_filter=self.n_convfilter[3], kernel_size=3, strides=(1,1,1), dilation_rate = (1,2,2),
                                      use_bias=True, is_separable=self.is_separable)
        self.conv4e = SeparableConv3D(conv_filter=self.n_convfilter[3], kernel_size=3, strides=(1,1,1), dilation_rate = (1,2,2),
                                      use_bias=True, is_separable=self.is_separable)

        
    def call(self, x):
        # Down-residual block 1
        bn1a = self.bn1a(x)
        
        relu1a = self.relu(bn1a)
        conv1a = self.conv1a(relu1a)
        bn1b = self.bn1b(conv1a)
        relu1b = self.relu(bn1b)
        conv1b = self.conv1b(relu1b)
        conv1c = self.conv1c(x)
        add1a = Add()([conv1b, conv1c])
        
        bn1c = self.bn1c(add1a)
        relu1c = self.relu(bn1c)
        conv1d = self.conv1d(relu1c)
        bn1d = self.bn1d(conv1d)
        relu1d = self.relu(bn1d)
        conv1e = self.conv1e(relu1d)
        add1b = Add()([add1a, conv1e])
        d1_out = self.drop(add1b)
        
        
        # Down-residual block 2
        bn2a = self.bn2a(d1_out)
        
        relu2a = self.relu(bn2a)
        conv2a = self.conv2a(relu2a)
        bn2b = self.bn2b(conv2a)
        relu2b = self.relu(bn2b)
        conv2b = self.conv2b(relu2b)
        conv2c = self.conv2c(d1_out)
        add2a = Add()([conv2b, conv2c])
        
        bn2c = self.bn2c(add2a)
        relu2c = self.relu(bn2c)
        conv2d = self.conv2d(relu2c)
        bn2d = self.bn2d(conv2d)
        relu2d = self.relu(bn2d)
        conv2e = self.conv2e(relu2d)
        add2b = Add()([add2a, conv2e])
        d2_out = self.drop(add2b)
        
        
        # Down-residual block 3
        bn3a = self.bn3a(d2_out)
        
        relu3a = self.relu(bn3a)
        conv3a = self.conv3a(relu3a)
        bn3b = self.bn3b(conv3a)
        relu3b = self.relu(bn3b)
        conv3b = self.conv3b(relu3b)
        conv3c = self.conv3c(d2_out)
        add3a = Add()([conv3b, conv3c])
        
        bn3c = self.bn3c(add3a)
        relu3c = self.relu(bn3c)
        conv3d = self.conv3d(relu3c)
        bn3d = self.bn3d(conv3d)
        relu3d = self.relu(bn3d)
        conv3e = self.conv3e(relu3d)
        add3b = Add()([add3a, conv3e])
        d3_out = self.drop(add3b)
        
        
        # Down-residual block 4
        bn4a = self.bn4a(d3_out)
        
        relu4a = self.relu(bn4a)
        conv4a = self.conv4a(relu4a)
        bn4b = self.bn4b(conv4a)
        relu4b = self.relu(bn4b)
        conv4b = self.conv4b(relu4b)
        conv4c = self.conv4c(d3_out)
        add4a = Add()([conv4b, conv4c])
        
        bn4c = self.bn4c(add4a)
        relu4c = self.relu(bn4c)
        conv4d = self.conv4d(relu4c)
        bn4d = self.bn4d(conv4d)
        relu4d = self.relu(bn4d)
        conv4e = self.conv4e(relu4d)
        add4b = Add()([add4a, conv4e])
        
        d4_out = self.drop(add4b)
        
        return d4_out


class decoder(Layer):
    def __init__(self, n_deconvfilter, is_separable, reg=1e-4, dropout_rate=2e-2):
        super(decoder, self).__init__()
        
        self.n_deconvfilter = n_deconvfilter
        self.is_separable = is_separable
        
        self.relu = Activation('relu')
        self.drop = Dropout(dropout_rate)
        
        self.bn5a = BatchNormalization()
        self.bn5b = BatchNormalization()
        self.bn5c = BatchNormalization()
        self.bn5d = BatchNormalization()
        self.ct5a = SeparableConv3DTranspose(conv_filter=n_deconvfilter[0], kernel_size=3, strides=(1,2,2), dilation_rate=(1,1,1),
                                             use_bias=True, is_separable=self.is_separable)
        self.conv5a = SeparableConv3D(conv_filter=n_deconvfilter[1], kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1),
                                      use_bias=True, is_separable=self.is_separable)
        self.ct5b = SeparableConv3DTranspose(conv_filter=n_deconvfilter[1], kernel_size=2, strides=(1,2,2), dilation_rate=(1,1,1),
                                             use_bias=True, is_separable=self.is_separable)
        self.conv5b = SeparableConv3D(conv_filter=n_deconvfilter[1], kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1),
                                      use_bias=True, is_separable=self.is_separable)
        self.conv5c = SeparableConv3D(conv_filter=n_deconvfilter[1], kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1),
                                      use_bias=True, is_separable=self.is_separable)        
        
        self.bn6a = BatchNormalization()
        self.bn6b = BatchNormalization()
        self.bn6c = BatchNormalization()
        self.bn6d = BatchNormalization()
        self.ct6a = SeparableConv3DTranspose(conv_filter=n_deconvfilter[2], kernel_size=3, strides=(1,2,2), dilation_rate=(1,1,1),
                                            use_bias=True, is_separable=self.is_separable)
        self.conv6a = SeparableConv3D(conv_filter=n_deconvfilter[2], kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1),
                                     use_bias=True, is_separable=self.is_separable)
        self.ct6b = SeparableConv3DTranspose(conv_filter=n_deconvfilter[2], kernel_size=2, strides=(1,2,2), dilation_rate=(1,1,1),
                                            use_bias=True, is_separable=self.is_separable)
        self.conv6b = SeparableConv3D(conv_filter=n_deconvfilter[2], kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1),
                                     use_bias=True, is_separable=self.is_separable)
        self.conv6c = SeparableConv3D(conv_filter=n_deconvfilter[2], kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1),
                                     use_bias=True, is_separable=self.is_separable)         
        
        self.bn7a = BatchNormalization()
        self.bn7b = BatchNormalization()
        self.bn7c = BatchNormalization()
        self.bn7d = BatchNormalization()
        self.ct7a = SeparableConv3DTranspose(conv_filter=n_deconvfilter[3], kernel_size=3, strides=(1,2,2), dilation_rate=(1,1,1),
                                             use_bias=True, is_separable=self.is_separable)
        self.conv7a = SeparableConv3D(conv_filter=n_deconvfilter[3], kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1),
                                      use_bias=True, is_separable=self.is_separable)
        self.ct7b = SeparableConv3DTranspose(conv_filter=n_deconvfilter[3], kernel_size=2, strides=(1,2,2), dilation_rate=(1,1,1),
                                             use_bias=True, is_separable=self.is_separable)
        self.conv7b = SeparableConv3D(conv_filter=n_deconvfilter[3], kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1),
                                      use_bias=True, is_separable=self.is_separable)
        self.conv7c = SeparableConv3D(conv_filter=n_deconvfilter[3], kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1),
                                      use_bias=True, is_separable=self.is_separable)        
        
        self.bn8a = BatchNormalization()
        self.bn8b = BatchNormalization()
        self.bn8c = BatchNormalization()
        self.bn8d = BatchNormalization()
        self.ct8a = SeparableConv3DTranspose(conv_filter=n_deconvfilter[4], kernel_size=3, strides=(1,2,2), dilation_rate=(1,1,1),
                                             use_bias=True, is_separable=self.is_separable)
        self.conv8a = SeparableConv3D(conv_filter=n_deconvfilter[4], kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1),
                                      use_bias=True, is_separable=self.is_separable)
        self.ct8b = SeparableConv3DTranspose(conv_filter=n_deconvfilter[4], kernel_size=2, strides=(1,2,2), dilation_rate=(1,1,1),
                                             use_bias=True, is_separable=self.is_separable)
        self.conv8b = SeparableConv3D(conv_filter=n_deconvfilter[4], kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1),
                                      use_bias=True, is_separable=self.is_separable)
        self.conv8c = SeparableConv3D(conv_filter=n_deconvfilter[4], kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1),
                                      use_bias=True, is_separable=self.is_separable) 

        self.bn9a = BatchNormalization()
        self.bn9b = BatchNormalization()
        self.bn9c = BatchNormalization()
        self.bn9d = BatchNormalization()
        self.conv9a = SeparableConv3D(conv_filter=n_deconvfilter[4], kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1),
                                      use_bias=True, is_separable=self.is_separable)
        self.conv9b = SeparableConv3D(conv_filter=n_deconvfilter[5], kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1),
                                      use_bias=True, is_separable=self.is_separable)
        self.conv9c = SeparableConv3D(conv_filter=n_deconvfilter[5], kernel_size=1, strides=(1,1,1), dilation_rate=(1,1,1),
                                      use_bias=True, is_separable=self.is_separable)
        self.conv9d = SeparableConv3D(conv_filter=n_deconvfilter[5], kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1),
                                      use_bias=True, is_separable=self.is_separable)
        self.conv9e = SeparableConv3D(conv_filter=n_deconvfilter[5], kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1),
                                      use_bias=True, is_separable=self.is_separable)
        self.conv9f = SeparableConv3D(conv_filter=n_deconvfilter[5], kernel_size=1, strides=(1,1,1), dilation_rate=(1,1,1),
                                      use_bias=True, is_separable=self.is_separable)        
        
        self.bn10a = BatchNormalization()
        self.bn10b = BatchNormalization()
        self.bn10c = BatchNormalization()
        self.bn10d = BatchNormalization()
        self.conv10a = SeparableConv3D(conv_filter=n_deconvfilter[6], kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1),
                                       use_bias=True, is_separable=self.is_separable)
        self.conv10b = SeparableConv3D(conv_filter=n_deconvfilter[6], kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1),
                                       use_bias=True, is_separable=self.is_separable)
        self.conv10c = SeparableConv3D(conv_filter=n_deconvfilter[6], kernel_size=1, strides=(1,1,1), dilation_rate=(1,1,1),
                                       use_bias=True, is_separable=self.is_separable)
        self.conv10d = SeparableConv3D(conv_filter=n_deconvfilter[6], kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1),
                                       use_bias=True, is_separable=self.is_separable)
        self.conv10e = SeparableConv3D(conv_filter=n_deconvfilter[6], kernel_size=3, strides=(1,1,1), dilation_rate=(1,1,1),
                                       use_bias=True, is_separable=self.is_separable)
        self.conv10f = SeparableConv3D(conv_filter=n_deconvfilter[6], kernel_size=1, strides=(1,1,1), dilation_rate=(1,1,1),
                                       use_bias=True, is_separable=self.is_separable)  
        
        
    def call(self, x):       
        # Up-residual block 1
        bn5a = self.bn5a(x)
        relu5a = self.relu(bn5a)
        ct5a = self.ct5a(relu5a)
        bn5b = self.bn5b(ct5a)
        relu5b = self.relu(bn5b)
        conv5a = self.conv5a(relu5b)
        ct5b = self.ct5b(x)
        add5a = Add()([conv5a, ct5b])
        
        bn5c = self.bn5c(add5a)
        relu5c = self.relu(bn5c)
        conv5b = self.conv5b(relu5c)
        bn5d = self.bn5d(conv5b)
        relu5d = self.relu(bn5d)
        conv5c = self.conv5c(relu5d)
        add5b = Add()([conv5c, add5a])
        
        # u1_out = Concatenate()([add5b, d3_out])
        u1_out = self.drop(add5b)


        # Up-residual block 2
        bn6a = self.bn6a(u1_out)
        relu6a = self.relu(bn6a)
        ct6a = self.ct6a(relu6a)
        bn6b = self.bn6b(ct6a)
        relu6b = self.relu(bn6b)
        conv6a = self.conv6a(relu6b)
        ct6b = self.ct6b(u1_out)
        add6a = Add()([conv6a, ct6b])
        
        bn6c = self.bn6c(add6a)
        relu6c = self.relu(bn6c)
        conv6b = self.conv6b(relu6c)
        bn6d = self.bn6d(conv6b)
        relu6d = self.relu(bn6d)
        conv6c = self.conv6c(relu6d)
        
        add6b = Add()([add6a, conv6c])
        # u2_out = Concatenate()([add6b, d2_out])
        u2_out = self.drop(add6b)
        
        
        # Up-residual block 3
        bn7a = self.bn7a(u2_out)
        relu7a = self.relu(bn7a)
        ct7a = self.ct7a(relu7a)
        bn7b = self.bn7b(ct7a)
        relu7b = self.relu(bn7b)
        conv7a = self.conv7a(relu7b)
        ct7b = self.ct7b(u2_out)
        add7a = Add()([conv7a, ct7b])
        
        bn7c = self.bn7c(add7a)
        relu7c = self.relu(bn7c)
        conv7b = self.conv7b(relu7c)
        bn7d = self.bn7d(conv7b)
        relu7d = self.relu(bn7d)
        conv7c = self.conv7c(relu7d)
        
        add7b = Add()([add7a, conv7c])
        # u3_out = Concatenate()([add7b, d1_out])
        u3_out = self.drop(add7b)
        
        
        # Up-residual block 4
        bn8a = self.bn8a(u3_out)
        relu8a = self.relu(bn8a)
        ct8a = self.ct8a(relu8a)
        bn8b = self.bn8b(ct8a)
        relu8b = self.relu(bn8b)
        conv8a = self.conv8a(relu8b)
        ct8b = self.ct8b(u3_out)
        add8a = Add()([conv8a, ct8b])
        
        bn8c = self.bn8c(add8a)
        relu8c = self.relu(bn8c)
        conv8b = self.conv8b(relu8c)
        bn8d = self.bn8d(conv8b)
        relu8d = self.relu(bn8d)
        conv8c = self.conv8c(relu8d)
        
        add8b = Add()([add8a, conv8c])
        # u4_out = Concatenate()([add8b, x])
        u4_out = self.drop(add8b)
        
        
        # Residual block 1
        bn9a = self.bn9a(u4_out)
        relu9a = self.relu(bn9a)
        conv9a = self.conv9a(relu9a)
        bn9b = self.bn9b(conv9a)
        relu9b = self.relu(bn9b)
        conv9b = self.conv9b(relu9b)
        conv9c = self.conv9c(u4_out)
        add9a = Add()([conv9b, conv9c])
        
        bn9c = self.bn9c(add9a)
        relu9c = self.relu(bn9c)
        conv9d = self.conv9d(relu9c)
        bn9d = self.bn9d(conv9d)
        relu9d = self.relu(bn9d)
        conv9e = self.conv9e(relu9d)
        conv9f = self.conv9f(add9a)
        add9b = Add()([conv9e, conv9f])
        r1_out = self.drop(add9b)
        
        
        # Residual block 2
        bn10a = self.bn10a(r1_out)
        relu10a = self.relu(bn10a)
        conv10a = self.conv10a(relu10a)
        bn10b = self.bn10b(conv10a)
        relu10b = self.relu(bn10b)
        conv10b = self.conv10b(relu10b)
        conv10c = self.conv10c(r1_out)
        add10a = Add()([conv10b, conv10c])
        
        bn10c = self.bn10c(add10a)
        relu10c = self.relu(bn10c)
        conv10d = self.conv10d(relu10c)
        bn10d = self.bn10d(conv10d)
        relu10d = self.relu(bn10d)
        conv10e = self.conv10e(relu10d)
        conv10f = self.conv10f(add10a)
        add10b = Add()([conv10e, conv10f])
        r2_out = self.drop(add10b)

        return r2_out

