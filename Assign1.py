# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 12:27:33 2019

@author: Vishal Kapur
"""

import pandas as pd


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
#from keras.utils import np_utils

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

tf.random.set_seed(100)
mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

num_train, img_rows, img_cols =  train_images.shape
num_test, _, _ =  test_images.shape
num_classes = len(np.unique(train_labels))
print(train_images.shape)
print(test_images.shape)
#(60000, 28, 28)
#(10000, 28, 28)


train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(10,activation='softmax')
        ])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

print(model.summary())
history = model.fit(train_images, train_labels, batch_size=512, validation_data = (test_images, test_labels), epochs=10)

#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#flatten_6 (Flatten)          (None, 784)               0         
#_________________________________________________________________
#dense_6 (Dense)              (None, 10)                7850      
#=================================================================
#Total params: 7,850
#Trainable params: 7,850
#Non-trainable params: 0
#Epoch 9/10
#60000/60000 [==============================] - 0s 7us/sample - loss: 0.3079 - accuracy: 0.9158 - val_loss: 0.2954 - val_accuracy: 0.9185
#Epoch 10/10
#60000/60000 [==============================] - 0s 6us/sample - loss: 0.3007 - accuracy: 0.9179 - val_loss: 0.2902 - val_accuracy: 0.9180

  

model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(300,activation='relu'),
        keras.layers.Dense(10,activation='softmax')
        ])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

print(model.summary())
history = model.fit(train_images, train_labels, batch_size=512, validation_data = (test_images, test_labels), epochs=10)



#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#flatten_7 (Flatten)          (None, 784)               0         
#_________________________________________________________________
#dense_7 (Dense)              (None, 300)               235500    
#_________________________________________________________________
#dense_8 (Dense)              (None, 10)                3010      
#=================================================================
#Total params: 238,510
#Trainable params: 238,510
#Non-trainable params: 0
#Epoch 9/10
#60000/60000 [==============================] - 1s 12us/sample - loss: 0.0491 - accuracy: 0.9866 - val_loss: 0.0720 - val_accuracy: 0.9779
#Epoch 10/10
#60000/60000 [==============================] - 1s 12us/sample - loss: 0.0418 - accuracy: 0.9892 - val_loss: 0.0713 - val_accuracy: 0.9770


