import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf

import cv2
import os

import numpy as np

strikers = ['Dubu', 'Rune']
model = tf.keras.models.load_model('models/{}-{}.h5'.format(strikers[0], strikers[1]))

img_size = 224

def get_data(data_dir, labels):
  data = [] 
  for label in labels: 
    path = os.path.join(data_dir, label)
    class_num = labels.index(label)
    for img in os.listdir(path):
      try:
        img_arr = cv2.imread(os.path.join(path, img))[...,::-1] 
        resized_arr = cv2.resize(img_arr, (img_size, img_size)) 
        data.append([resized_arr, class_num])
      except Exception as e:
        print(e)
  return np.array(data)

train = get_data('./input/train', strikers)
val = get_data('./input/test', strikers)

x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
  x_train.append(feature)
  y_train.append(label)

for feature, label in val:
  x_val.append(feature)
  y_val.append(label)

x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

print('\n\n')

pred_train= model.predict(x_train)
scores = model.evaluate(x_train, y_train, verbose=0)
print('Accuracy on training data: {}'.format(scores[1]))   

pred_test= model.predict(x_val)
scores2 = model.evaluate(x_val, y_val, verbose=0)
print('Accuracy on test data: {}'.format(scores2[1]))