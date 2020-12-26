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
path = "/volumes/RolanG/Tests/"

# get images and labels

for folder in os.listdir(path):
    if not folder.startswith('.'):
        for file in os.listdir(path+folder):
            image=cv2.imread(path+folder+"/"+file,cv2.COLOR_BGR2HSV) # converting image to HSV
            if image is not None:
                image=image.astype("float32")
                resized_image=cv2.resize(image,(270,480))
                img.append(resized_image)
                if folder=="threepics":
                    labels.append(2)
                if folder=="onepics":
                    labels.append(1)
                if folder=="zeropics":
                    labels.append(0)

#print(img)
#print(labels)

# works!
# train test split
lenImgs=len(img)

train_images=np.array(img[int(lenImgs/5):])
test_images=np.array(img[:int(lenImgs/5)])
train_labels=np.array(labels[int(lenImgs/5):])
test_labels=np.array(labels[:int(lenImgs/5)])

# print(len(train_images))
# print(len(test_images))
#
# for img in train_images:
#     print("shape: ", img.shape)
#print("shape: ", test_images[0].shape)

# train-test split works
# time to create model

class_names=['three','one','zero']
num_classes=len(class_names)

# new model's sequential
# model = Sequential([
#   layers.experimental.preprocessing.Rescaling(1./255, input_shape=(480, 270, 3)),
#   # layers.Conv2D(16, 3, padding='same', activation='relu'),
#   # layers.MaxPooling2D(),
#   # layers.Conv2D(32, 3, padding='same', activation='relu'),
#   # layers.MaxPooling2D(),
#   # layers.Conv2D(64, 3, padding='same', activation='relu'),
#   # layers.MaxPooling2D(),
#   layers.Flatten(),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(num_classes)
# ])

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(32,activation='relu'),
#  tf.keras.layers.Dense(192, activation='relu'), # have to be 192?

  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(3)
])

# new model's compile
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)

# model.summary()
# old model.fit
# epochs=10
#
# history = model.fit(
#   train_images,
#   validation_data=train_labels,
#   epochs=epochs
# )

model.fit(
    np.array(train_images),
    np.array(train_labels),
    epochs=6,
)

results = model.evaluate(test_images,  test_labels)

print(results)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
