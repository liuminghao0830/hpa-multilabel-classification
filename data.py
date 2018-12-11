# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
from keras import backend as K
from sklearn.model_selection import StratifiedKFold


K.set_image_data_format('channels_last')


def read_train_labels(file):
  data = []
  print('Reading labels ...')
  with open(file) as f:
    lis = [line.strip('\n').split(',') for line in f]
    for i,x in enumerate(lis):
      if i == 0:
        continue
      data.append(x)
  return data


def class_weight(labels):
  labels = read_train_labels(label_path)
  y_train = [np.array(x[1].split()).astype(np.int) for x in labels]
  y_count = []
  for y in y_train:
    y_count.extend(y)
  cw = compute_class_weight('balanced', np.arange(28), y_count)
  return cw


def kfold_dataset(labels, weights, n_folds):
  label_list = [np.array(x[1].split(),dtype=np.int) for x in labels]
  y = np.array([x[np.argmax(weights[x])] for x in label_list])
  X = np.arange(len(labels))
  skf = StratifiedKFold(n_splits=n_folds)
  return skf.split(X, y)




class DataGenerator():
    'Generates data for Keras'
    def __init__(self, path, list_IDs, labels, batch_size=32, dim=(224,224), 
                 n_channels=3, n_classes=28, augment=True, shuffle=True):
        'Initialization'
        self.data_path = path
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.augment = augment
        self.shuffle = shuffle


    def create_dataset(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        def create_target(x):
            target = np.array(x.split()).astype(np.int)
            one_hot = np.zeros(28).astype(np.int)
            one_hot[target] = 1
            return one_hot
       
        while True:
            # Find list of IDs
            list_IDs_temp = np.random.choice(len(self.list_IDs), self.batch_size)
            
            # Initialization
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
            y = np.empty((self.batch_size, self.n_classes), dtype=int)
        
            # Generate data
            colors = ['red', 'green', 'blue']
            for i, ID in enumerate(list_IDs_temp):                                  
                # Read image and store in batch                                     
                img_name = self.labels[ID][0]                                       
                image_array = np.zeros((self.dim[0],self.dim[1],4))                
                for n, c in enumerate(colors):                                      
                    im = Image.open(self.data_path + '/' + img_name + '_' + c + '.png')
                    im_res = im.resize(self.dim, Image.ANTIALIAS)                   
                    image_array[:,:,n] = np.array(im_res.getdata()).reshape((self.dim))
                if self.augment:                                                    
                    image_array = DataGenerator.augment(image_array[:,:,:self.n_channels])
                    X[i,:,:,:] = image_array / 255.                                 
                else:                                                               
                    X[i,:,:,:] = image_array[:,:,:self.n_channels] / 255.           
                im.close()                                                          
                                                                                
                # Store class
                y[i,:] = create_target(self.labels[ID][1])
            yield X, y

    def augment(image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug

