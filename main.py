# -*- coding: utf-8 -*-
"""
Train base CNN model for image classification of Kaggle HPA competition.
Base models: DenseNet121, Inceptionv3, InceptionResNetv2

Author: Minghao Liu
Affliate: Mechanical Engineering, Arizona State University
Date: Dec/13/2018
"""


import numpy as np
import keras
from keras.models import *
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam 
from keras import backend as K
from keras.utils import multi_gpu_model

import data
import model
import utils

K.set_image_data_format('channels_last')


# Start training model
input_shape = (512, 512, 3)

keras.backend.clear_session()

train_model = model.create_model(input_shape=input_shape, n_out=28)

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.03)

train_model.compile(loss=model.combine_loss, optimizer=adam, metrics=['acc', model.f1])

train_model.summary()


# Parameters
params = {'dim': (input_shape[0],input_shape[1]),
          'batch_size': 32,
          'n_classes': 28,
          'n_channels': input_shape[2],
          'augment':True,
          'shuffle': False}

# Datasets
train_data_path = '../../train'
label_path = '../../train.csv'

labels = data.read_train_labels(label_path)

loss_weight = utils.class_weight(labels)

n_gpu = 2
n_epochs = 30
k_fold = False

if k_fold:
    train_data_set = data.kfold_dataset(labels, loss_weight, n_folds=5)
else:
    # split and suffle data 
    np.random.seed(2018)
    indexes = np.arange(len(labels))
    np.random.shuffle(indexes)
    train_indexes = indexes[:25500]
    valid_indexes = indexes[25500:]
    train_data_set = [[train_indexes, valid_indexes]]


for i, (train_indexes, test_indexes) in enumerate(train_data_set):
    print("Running Fold", i+1)
    
    # Generators
    training_generator = data.DataGenerator(train_data_path, train_indexes, labels, **params)
    validation_generator = data.DataGenerator(train_data_path, test_indexes, labels, **params)

    checkpoint = ModelCheckpoint('best_val_f1.h5', monitor='val_f1', verbose=1, save_best_only=True, save_weights_only=True, mode='max', period=1)

    #train_model.load_weights('weights/inceptionv3_lb0443.h5')

    # train model                                                                   
    if n_gpu > 1:                                                                   
        parallel_model = multi_gpu_model(train_model, gpus=n_gpu)                         
        parallel_model.compile(loss=model.combine_loss, optimizer=adam, metrics=['acc', model.f1])
        parallel_model.fit_generator(generator=training_generator.create_dataset(), 
                            validation_data=next(validation_generator.create_dataset()),
                            epochs = n_epochs,                                      
                            steps_per_epoch=np.floor(len(train_indexes)/params['batch_size']),
                            class_weight=loss_weight,                               
                            use_multiprocessing=False,                              
                            workers=1,                                              
                            callbacks=[checkpoint],                                 
                            verbose=1)                                              
    else:                                                                           
        train_model.fit_generator(generator=training_generator.create_dataset(),          
                            validation_data=next(validation_generator.create_dataset()),
                            epochs = n_epochs,                                      
                            steps_per_epoch=np.floor(len(train_indexes)/params['batch_size']),   
                            class_weight=loss_weight,                               
                            use_multiprocessing=False,                              
                            workers=1,
                            callbacks=[checkpoint],                                              
                            verbose=1)

train_model.save('inceptionv3_test.h5')

