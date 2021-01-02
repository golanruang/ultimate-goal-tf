import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join, expanduser
import os
import subprocess
import random

import cv2
import PIL

img=[]
labels=[]
path = "/volumes/RolanG/ftc-tensorflow-data/"
#path = "/volumes/RolanG/Tests/"

# get images and labels

for folder in os.listdir(path):
    if not folder.startswith('.'):
        for file in os.listdir(path+folder):
            image=cv2.imread(path+folder+"/"+file,cv2.COLOR_BGR2HSV) # converting image to HSV
            if image is not None:
                image=image.astype("float32")
                # resized to squre because algorithms are optimized for squares
                # could also be (256,256)
                resized_image=cv2.resize(image,(231,231))
                img.append(resized_image)
                if folder=="threepics":
                    labels.append("2")
                if folder=="onepics":
                    labels.append("1")
                if folder=="zeropics":
                    labels.append("0")

# works!
# train test split

for i, pic in enumerate(img):
    for j, row in enumerate(pic):
        for k, pixel in enumerate(row):
            for l, color_value in enumerate(pixel):
                img[i][j][k][l] = color_value / 255.0

lenImgs=len(img)

# import random + shuffle them
train_images=np.array(img[int(lenImgs/5):])
test_images=np.array(img[:int(lenImgs/5)])
train_labels=np.array(labels[int(lenImgs/5):])
test_labels=np.array(labels[:int(lenImgs/5)])

# train-test split works
# time to create model

  # relu: any negative output gets put to 0
  # if model works pretty well, probably switch to elu

# might want to preprocess Hue by dividing by 180
class_names=['3','1','0']
num_classes=len(class_names)

img_height=231
img_width=231

scale = 1./255

model = Sequential([
  # layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  # first num is # filters, kernel size 11x11,
  layers.Conv2D(16, 11, strides=(4,4), padding='valid', activation='relu'),
  # 56x56x16 output
  layers.MaxPooling2D(),
  # Max pool shrinks output by half
  # 28x28x16
  layers.Conv2D(32, 3, strides=(1,1), padding='valid', activation='relu'),
  # 26x26x32
  layers.MaxPooling2D(),
  # 13x13x32
  layers.Conv2D(64, 5, strides=(1,1), padding='valid', activation='relu'),
  # 9x9x64
  layers.MaxPooling2D(pool_size=(3,3)),
  # 3x3x64
  layers.Flatten(),
  # 576 vector
  layers.Dense(128, activation='relu'),
  # 128 vector
  layers.Dense(num_classes)
  # 3 vector
])

# new model's compile
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# model.compile(
#     loss= tf.keras.losses.SparseCategoricalCrossentropy(), # look this up (means you're using a loss calculation)
#     optimizer=tf.keras.optimizers.Adam(0.001),
#     metrics=[tf.keras.metrics.Accuracy()],                   # training for accuracy on training set rather than loss on training set (good)
# )
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Things to do to increase accuracy:
# (if good model already) --> switch from relu to elu or leaky relu
# make sure at end of each epoch u capture loss on training set, accuracy on training set, accuracy on test/validation set
# graph those three things on same set of axises
# loss should be dropping asymptotically, accuracy (training) should increase towards 1, accuracy on test set should increase for a while,
# then start to decrease eventually. When switched from increasing to decreasing, you're overfitting and you stop training there

model.fit(
    np.array(train_images),
    np.array(train_labels),
    epochs=12,
    validation_data=(test_images,test_labels)
)

# print("train results: ")
# results = model.evaluate(train_images,  train_labels)
#
# print(results)
#
# print("test results: ")
#
# results = model.evaluate(test_images,  test_labels)
#
# print(results)

tf.keras.models.save_model(model, 'RGModel.h5')

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
#
# # Save the model.
# with open('model.tflite', 'wb') as f:
#   f.write(tflite_model)
