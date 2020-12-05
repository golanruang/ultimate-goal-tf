import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join, expanduser
import os
import subprocess
import random

import cv2

path = "/volumes/RGUSB/ftc-tensorflow-data"
img=[]
labels=[]
for folder in os.listdir(path):
    if not folder.startswith('.'):
        if folder=="threepics":
            for file in os.listdir(path+"/threepics"):
                print(file)
                print(cv2.imread(path+"/threepics/"+file))
                img.append(cv2.imread(path+"threepics/"+file))
                labels.append("three")
        if folder=="onepics":
            for file in os.listdir(path+"/onepics"):
                img.append(cv2.imread(path+"/onepics/"+file))
                labels.append("one")
        if folder=="zeropics":
            for file in os.listdir(path+"/zeropics"):
                img.append(cv2.imread(path+"/zeropics/"+file))
                labels.append("zero")

#print(img)
#print(labels)
#print(img)
data=list(zip(img,labels))

lenLists=len(data)

test=data[:int(lenLists/5)]
train=data[int(lenLists/5):]

testImgs=list(list(zip(*test))[0])
testLabels=list(list(zip(*test))[1])

trainImgs=list(list(zip(*train))[0])
trainLabels=list(list(zip(*test))[1])

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(128, 128)),
  tf.keras.layers.Dense(192, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(3)
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)

model.fit(
    trainImgs,
    epochs=6,
    validation_data=trainLabels,
)

results=model.evaluate(testImgs,  testLabels)
print(results)
