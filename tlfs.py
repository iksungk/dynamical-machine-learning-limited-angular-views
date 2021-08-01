#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras import backend as K

import scipy.io as sio
import tensorflow as tf
import numpy as np
import sys 
import h5py as hp


def g_loss_npcc(generated_image, true_image):
    fsp=generated_image-K.mean(generated_image,axis=(1,2,3,4),keepdims=True)
    fst=true_image-K.mean(true_image,axis=(1,2,3,4),keepdims=True)
    
    devP=K.std(generated_image,axis=(1,2,3,4))
    devT=K.std(true_image,axis=(1,2,3,4))
    
    npcc_loss=(-1)*K.mean(fsp*fst,axis=(1,2,3,4))/K.clip(devP*devT,K.epsilon(),None)    ## (BL,1)
    return npcc_loss

