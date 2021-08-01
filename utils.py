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
import h5py as hp


class LearningRateBasedStopping(tf.keras.callbacks.Callback):
    def __init__(self, limit_lr):
        super(LearningRateBasedStopping, self).__init__()
        self.limit_lr = limit_lr
        
    def on_epoch_begin(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, lr))
                                    
        if lr < self.limit_lr:
            self.model.stop_training = True
            
            
class AngularAttention(Layer):
    def __init__(self):
        super(AngularAttention, self).__init__()
         
        self.reg = 1e-4
        self.Whe = Dense(1, activation=None, kernel_regularizer=regularizers.l2(self.reg), bias_regularizer=regularizers.l2(self.reg))
        self.tanh = Activation('tanh')
        self.softmax = Activation('softmax')
        
    def call(self, h):
        h_rsh = Reshape((h.shape[1], np.prod(h.shape[2:])))(h)
        eh = self.tanh(K.squeeze(self.Whe(h_rsh), axis=-1))
        
        e = Reshape([-1])(eh)
        alpha = self.softmax(e)
        alpha_ = alpha
        alpha = RepeatVector(np.prod(h.shape[2:]))(alpha)
        alpha = Reshape(h.shape[1:])(alpha)

        rec = K.sum(Multiply()([alpha, h]), axis=1, keepdims=False)
        
        return rec, alpha_
    

def apply_moving_window(x, sequence_length, option):
    N = x.shape[1]
    M = sequence_length
    L = N//2 - M//2 + 1 # window length
    
    tilde_x = []
    for m in range(M//2):
        if option == 'sum':
            tilde_x.append(tf.math.reduce_sum(x[:,m:m+L,:,:,:], axis=1))
            
        elif option == 'mean':
            tilde_x.append(tf.math.reduce_mean(x[:,m:m+L,:,:,:], axis=1))
            
    for m in range(M//2):
        m_ = m + N//2
        if option == 'sum':
            tilde_x.append(tf.math.reduce_sum(x[:,m_:m_+L,:,:,:], axis=1))
            
        elif option == 'mean':
            tilde_x.append(tf.math.reduce_mean(x[:,m_:m_+L,:,:,:], axis=1))
    
    tilde_x = K.permute_dimensions(tf.stack(tilde_x), (1,0,2,3,4,5))
    return tilde_x

