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

data = ['AiMi', 'Asher', 'Drekar', 'Dubu', 'Estelle', 'Juliette', 'Kai', 'Rune']
# striker_labels = ['AiMi', 'Asher', 'Atlas', 'Drekar', 'Dubu', 'Era', 'Estelle', 'Juliette', 'Juno', 'Kai', 'Luna', 'Rune', 'X']
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

def gen_model(strikers):

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

  model = Sequential()
  model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(224,224,3)))
  model.add(MaxPool2D())

  model.add(Conv2D(32, 3, padding="same", activation="relu"))
  model.add(MaxPool2D())

  model.add(Conv2D(64, 3, padding="same", activation="relu"))
  model.add(MaxPool2D())
  model.add(Dropout(0.4))

  model.add(Flatten())
  model.add(Dense(128,activation="relu"))
  model.add(Dense(2, activation="softmax"))

  opt = Adam(learning_rate=0.0001)
  model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])
  history = model.fit(x_train,y_train,epochs = 10 , validation_data = (x_val, y_val))

  pred_train= model.predict(x_train)
  scores = model.evaluate(x_train, y_train, verbose=0)
  print('Accuracy on training data: {}'.format(scores[1]))   

  pred_test= model.predict(x_val)
  scores2 = model.evaluate(x_val, y_val, verbose=0)
  print('Accuracy on test data: {}'.format(scores2[1]))

  model.save('models/{}-{}'.format(strikers[0], strikers[1]))

for first in range(len(data)):
  for second in range(first+1, len(data)):
    print([data[first], data[second]])
    gen_model([data[first], data[second]])