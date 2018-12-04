# -*- coding: utf-8 -*-

import numpy as np
import keras
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K

K.set_image_data_format('channels_last')


def create_model(input_shape, n_out):
   
    pretrain_model = InceptionV3(
        include_top=False, pooling='avg',
        weights='imagenet', input_shape=input_shape)
    
    model = Sequential()
    model.add(pretrain_model)
    model.add(Dense(n_out))
    model.add(Activation('sigmoid'))

    return model    



THRESHOLD = 0.2
def f1(y_true, y_pred):
    #y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(true,pred):
    groundPositives = K.sum(true, axis=0) + K.epsilon()
    correctPositives = K.sum(true * pred, axis=0) + K.epsilon()
    predictedPositives = K.sum(pred, axis=0) + K.epsilon()

    precision = correctPositives / predictedPositives
    recall = correctPositives / groundPositives

    m = (2 * precision * recall) / (precision + recall)

    return 1-K.mean(m)

 
gamma = 2.0
epsilon = K.epsilon()
def focal_loss(y_true, y_pred):
    pt = y_pred * y_true + (1-y_pred) * (1-y_true)
    pt = K.clip(pt, epsilon, 1-epsilon)
    CE = -K.log(pt)
    FL = K.pow(1-pt, gamma) * CE
    loss = K.sum(FL, axis=1)
    return loss


# Combine f1 loss and binary crossentropy
def combine_loss(y_true, y_pred):
    loss = 0.5*keras.losses.binary_crossentropy(y_true, y_pred) + 0.5*f1_loss(y_true, y_pred)
    return loss

