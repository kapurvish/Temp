# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 21:03:43 2019

@author: Vishal Kapur
"""

import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(1)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
print("train_images shape:", train_images.shape)
print("test_images shape:", test_images.shape)
print("train_labels shape:", train_labels.shape)
print("test_labels shape:", test_labels.shape)


#train_images shape: (50000, 32, 32, 3)
#test_images shape: (10000, 32, 32, 3)
#train_labels shape: (50000, 1)
#test_labels shape: (10000, 1)

train_images = train_images / 255
test_images = test_images / 255

model =tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64,activation ='relu'))
model.add(tf.keras.layers.Dense(10,activation ='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_images, train_labels,epochs=5)


#Epoch 1/5
#50000/50000 [==============================] - 85s 2ms/sample - loss: 1.4620 - accuracy: 0.4756
#Epoch 2/5
#50000/50000 [==============================] - 81s 2ms/sample - loss: 1.0948 - accuracy: 0.6127
#Epoch 3/5
#50000/50000 [==============================] - 82s 2ms/sample - loss: 0.9132 - accuracy: 0.6785
#Epoch 4/5
#50000/50000 [==============================] - 82s 2ms/sample - loss: 0.7702 - accuracy: 0.7316
#Epoch 5/5
#50000/50000 [==============================] - 81s 2ms/sample - loss: 0.6434 - accuracy: 0.7762
#Model: "sequential_9"
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#conv2d_6 (Conv2D)            (None, 30, 30, 64)        1792      
#_________________________________________________________________
#conv2d_7 (Conv2D)            (None, 28, 28, 32)        18464     
#_________________________________________________________________
#flatten_2 (Flatten)          (None, 25088)             0         
#_________________________________________________________________
#dense_4 (Dense)              (None, 64)                1605696   
#_________________________________________________________________
#dense_5 (Dense)              (None, 10)                650       
#=================================================================
#Total params: 1,626,602
#Trainable params: 1,626,602
#Non-trainable params: 0

#Answer for Ques 1): should be 75-80 , C option.

print(model.summary())
