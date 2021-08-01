#!/usr/bin/env python
# coding: utf-8

# In[10]:


from __future__ import print_function, division, absolute_import

from tensorflow.keras.layers import Activation, Add, Dense, BatchNormalization, Concatenate, Dropout, Subtract, Flatten, Input, Lambda, Reshape
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, MaxPool3D, AveragePooling3D, UpSampling3D, ConvLSTM2D
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, AveragePooling2D, UpSampling2D
from tensorflow.keras.layers import Layer, RepeatVector, Permute, Multiply, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint, Callback, CSVLogger
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras import backend as K

from blocks import SeparableConvGRU3D, encoder, decoder
from utils import LearningRateBasedStopping, TemporalAttention_v2, apply_moving_window
from tlfs import g_loss_npcc

import scipy.io as sio
import tensorflow as tf
import numpy as np
import sys 
import h5py as hp
import math 
import argparse
import os

                                 
class ConvGRU3DNet(Layer):
    def __init__(self, 
                 num_tsteps, 
                 num_layers, 
                 num_rows, 
                 num_cols, 
                 num_channels, 
                 is_separable = True):
        super(ConvGRU3DNet, self).__init__()
        
        self.num_tsteps = num_tsteps
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.N_h = self.num_rows // 16 # self.N_h is always same as num_rows//(2^(# of pool in an encoder)).

        self.is_separable = is_separable
        self.n_convfilter = [24, 48, 96, 192]
        self.convgru3d_filter = [512]
        self.n_deconvfilter = [192, 192, 96, 48, 36, 24, 1]
        
        self.convgru3d = SeparableConvGRU3D(convgru3d_filter=self.convgru3d_filter, is_separable=self.is_separable)
        self.temporal_attention_end = TemporalAttention_v2()
        self.encoder = encoder(self.n_convfilter, self.is_separable)
        self.decoder = decoder(self.n_deconvfilter, self.is_separable)
        
        self.input_dropout_rate = 5e-2
        self.input_drop = Dropout(self.input_dropout_rate)

    def call(self, x):
        sequence_length = 12
        alpha = tf.ones(x.shape)
        x = apply_moving_window(Multiply()([x, alpha]), sequence_length, 'mean')

        h_forw = tf.zeros(shape=(x.shape[0], self.num_layers, self.N_h, self.N_h, self.convgru3d_filter[0])) 
        h_states_forw = []
        dec_outputs = []
        
        for k in range(x.shape[1]):
            x_forw = x[:,k,:,:,:,:]
            x_enc_forw = self.encoder(x_forw)
            x_enc_forw = self.input_drop(x_enc_forw)
            h_forw = self.convgru3d(x_enc_forw, h_forw)
            h_forw_drop = self.input_drop(h_forw)
            h_states_forw.append(h_forw_drop)    
        
        h_gru_bf_att = K.permute_dimensions(tf.stack(h_states_forw), pattern=(1,0,2,3,4,5))
            
        # tf.stack(h_states): N_view x batch_size x num_layers x num_rows//16 x num_cols//16 x n_convgru3d_filter[0]
        # -> h_gru_bf_att: batch_size x N_view x num_layers x num_rows//16 x num_cols//16 x n_convgru3d_filter[0]; batch axis: 0.
                                             
        h_att, _ = self.temporal_attention_end(h_gru_bf_att)
        rec = self.decoder(h_att)


    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_tsteps': self.num_tsteps,
            'num_layers': self.num_layers,
            'num_rows': self.num_rows,
            'num_cols': self.num_cols,
            'num_channels': self.num_channels,
            'is_separable': self.is_separable
        })
        return config


is_separable = True
order_of_approximants = 'normal' #'normal', 'random', 'symmetric'

batch_size = 10
num_epochs = 200
num_tsteps = 42
num_layers = 4
num_rows = 64
num_cols = 64
num_channels = 1
                                             
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    x_in = Input(shape=(num_tsteps, num_layers, num_rows, num_cols, num_channels), batch_size=batch_size)
    out = ConvGRU3DNet(num_tsteps, num_layers, num_rows, num_cols, num_channels, is_separable)(x_in)
    convgru3d_model = Model(x_in, out)
    convgru3d_model.summary()

    optadam = optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    convgru3d_model.compile(optimizer=optadam, loss=g_loss_npcc, metrics=[g_loss_npcc, 'mse'])

