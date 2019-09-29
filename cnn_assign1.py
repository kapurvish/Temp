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
model.fit(train_images, train_labels,epochs=5,batch_size=128, shuffle=True)


#50000/50000 [==============================] - 67s 1ms/sample - loss: 1.5287 - accuracy: 0.4519
#Epoch 2/5
#50000/50000 [==============================] - 65s 1ms/sample - loss: 1.1510 - accuracy: 0.5970
#Epoch 3/5
#50000/50000 [==============================] - 64s 1ms/sample - loss: 0.9651 - accuracy: 0.6632
#Epoch 4/5
#50000/50000 [==============================] - 65s 1ms/sample - loss: 0.8449 - accuracy: 0.7071
#Epoch 5/5
#50000/50000 [==============================] - 65s 1ms/sample - loss: 0.7505 - accuracy: 0.7409

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

#Answer for Ques 1): accuracy : 0.7409, it should be 70-75 , B option.

print(model.summary())

model =tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64,activation ='relu'))
model.add(tf.keras.layers.Dense(10,activation ='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_images, train_labels,epochs=10,batch_size=128, shuffle=True)

#Model: "sequential_2"
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#conv2d_5 (Conv2D)            (None, 30, 30, 64)        1792      
#_________________________________________________________________
#max_pooling2d_2 (MaxPooling2 (None, 15, 15, 64)        0         
#_________________________________________________________________
#conv2d_6 (Conv2D)            (None, 13, 13, 32)        18464     
#_________________________________________________________________
#max_pooling2d_3 (MaxPooling2 (None, 6, 6, 32)          0         
#_________________________________________________________________
#flatten_2 (Flatten)          (None, 1152)              0         
#_________________________________________________________________
#dense_4 (Dense)              (None, 64)                73792     
#_________________________________________________________________
#dense_5 (Dense)              (None, 10)                650       
#=================================================================
#Total params: 94,698
#Trainable params: 94,698
#Non-trainable params: 0
#_________________________________________________________________
#None
#W0929 23:06:52.137400 11300 ag_logging.py:146] Entity <function Function._initialize_uninitialized_variables.<locals>.initialize_variables at 0x000002A2084B4BF8> could not be transformed and will be executed as-is. Please report this to the AutoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: 
#Train on 50000 samples
#Epoch 1/10
#WARNING: Entity <function Function._initialize_uninitialized_variables.<locals>.initialize_variables at 0x000002A2084B4BF8> could not be transformed and will be executed as-is. Please report this to the AutoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: 
#50000/50000 [==============================] - 42s 844us/sample - loss: 1.6467 - accuracy: 0.4093
#Epoch 2/10
#50000/50000 [==============================] - 39s 789us/sample - loss: 1.3197 - accuracy: 0.5303
#Epoch 3/10
#50000/50000 [==============================] - 39s 785us/sample - loss: 1.1893 - accuracy: 0.5813
#Epoch 4/10
#50000/50000 [==============================] - 39s 782us/sample - loss: 1.1068 - accuracy: 0.6102
#Epoch 5/10
#50000/50000 [==============================] - 39s 778us/sample - loss: 1.0488 - accuracy: 0.6332
#Epoch 6/10
#50000/50000 [==============================] - 41s 829us/sample - loss: 1.0013 - accuracy: 0.6516
#Epoch 7/10
#50000/50000 [==============================] - 42s 832us/sample - loss: 0.9695 - accuracy: 0.6622
#Epoch 8/10
#50000/50000 [==============================] - 40s 806us/sample - loss: 0.9372 - accuracy: 0.6752
#Epoch 9/10
#50000/50000 [==============================] - 40s 795us/sample - loss: 0.9100 - accuracy: 0.6842
#Epoch 10/10
#50000/50000 [==============================] - 49s 976us/sample - loss: 0.8864 - accuracy: 0.6923

#Answer for Ques 2 as accuracy: 0.6923 should be  65-70 , B option.
#Answer for Ques 3 should be  Reduced by 94,698 , b option.

