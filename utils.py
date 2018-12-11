# -*- coding: utf-8 -*-

import numpy as np
from sklearn.utils import compute_class_weight

import data


def class_weight(labels):
  y_train = [np.array(x[1].split()).astype(np.int) for x in labels]
  y_count = []
  for y in y_train:
    y_count.extend(y)
  cw = compute_class_weight('balanced', np.arange(28), y_count)
  return cw 

