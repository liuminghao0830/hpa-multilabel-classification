# -*- coding: utf-8 -*-
"""
Predict test image for Kaggle HPA competition.
Predicting methods: Average ensemble, Test Time Augmentation(TTA)

Author: Minghao Liu
Affliate: Mechanical Engineering, Arizona State University
Date: Dec/13/2018
"""

import numpy as np
from PIL import Image
import pandas as pd

import keras
from keras.models import Model, load_model, Input
from keras.layers import Average
from keras import backend as K

import model
from keras_tta import TTA_ModelWrapper

from tqdm import tqdm

K.set_image_data_format('channels_last')


def load_test_image(data_path, input_img_shape, idx):
    colors = ['red', 'green', 'blue']
    img = np.zeros(input_img_shape)
    for i, c in enumerate(colors):
        im = Image.open(data_path + '/' + idx + '_' + c + '.png')
        im_res = im.resize((input_img_shape[0], input_img_shape[1]), Image.ANTIALIAS)
        image_array = np.array(im_res.getdata()).reshape((input_img_shape[0], input_img_shape[1]))
        img[:,:,i] = image_array / 255.
        im.close()
    return np.array(img[:,:,:3])


def ensemble_predict(model_paths, input_shape, test_path, sample_submit, ensemble_weight):
    ensemble_weight = np.array(ensemble_weight) / np.sum(ensemble_weight)

    models = []
    for mp in model_paths:
        base_model = load_model(mp, custom_objects={'combine_loss':model.combine_loss, 'f1':model.f1})
        tta_model = TTA_ModelWrapper(base_model)
        models.append(base_model)
    predicted = []

    threshold = 0.3
    for name in tqdm(sample_submit['Id']):
        predict_prob = np.zeros(28)
        image = load_test_image(test_path, input_shape, name)
        for i,base_model in enumerate(models):
            predict_prob += ensemble_weight[i]*base_model.predict(image[np.newaxis])[0]
        #predict_prob = predict_prob / len(models)
        label_predict = np.arange(28)[predict_prob >= threshold]
        str_predict_label = ' '.join(str(l) for l in label_predict)
        predicted.append(str_predict_label)


    sample_submit['Predicted'] = predicted
    sample_submit.to_csv('submission.csv', index=False)





def base_model_predict(model_path, input_shape, test_path, sample_submit):
    predict_model = load_model(model_path, custom_objects={'combine_loss':model.combine_loss,'f1':model.f1})
    threshold = 0.3
    #model.load_weights('pretrained_model/inceptionv3_weights_60epoch.h5')

    tta_model = TTA_ModelWrapper(predict_model)

    predicted = []
    for name in tqdm(sample_submit['Id']):
        image = load_test_image(test_path, input_shape, name)
        score_predict = tta_model.predict(image[np.newaxis])[0]
        label_predict = np.arange(28)[score_predict[0] >= threshold]
        str_predict_label = ' '.join(str(l) for l in label_predict)
        predicted.append(str_predict_label)


    sample_submit['Predicted'] = predicted
    sample_submit.to_csv('submission.csv', index=False)


def main():
    ensemble = True

    input_img_shape = (512, 512, 3)

    threshold = 0.3

    submit = pd.read_csv('sample_submission.csv')

    test_path = 'test'

    #model_paths = ['pretrained_model/inceptionv3.h5','pretrained_model/densenet121.h5','pretrained_model/inceptionresnetv2.h5']
    model_paths = 'pretrained_model/inceptionv3_external.h5'
    if ensemble:
        ensemble_predict(model_paths, input_img_shape, test_path, submit, [0.45,0.45,0.42])
    else:
        base_model_predict(model_paths, input_img_shape, test_path, submit)


if __name__ == '__main__':
    main()
