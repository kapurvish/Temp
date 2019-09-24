# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 23:09:19 2019

@author: Vishal Kapur
"""

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import tensorflow as tf
import pandas as pd
from tensorflow import feature_column
from tensorflow import keras
pd.set_option('display.max_columns',15)
tf.random.set_seed(1)

boston_dataset = load_boston()

data_X = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
data_Y = pd.DataFrame(boston_dataset.target, columns=["target"])

#data_X.to_csv('data_X.csv')
#data_Y.to_csv('data_Y.csv')
               
data = pd.concat([data_X, data_Y], axis=1)
print(data.head())

train, test = train_test_split(data, test_size=0.2, random_state=1)
train, val = train_test_split(train, test_size=0.2, random_state=1)
print(len(train), "train examples")
print(len(val), "validation examples")
print(len(test), "test examples")

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

  
batch_size = 32
train_ds = df_to_dataset(train, True, batch_size)
val_ds = df_to_dataset(val, False, batch_size)
test_ds = df_to_dataset(test, False, batch_size)

#def get_scal(feature):
#  def minmax(x):
#    mini = train[feature].min()
#    maxi = train[feature].max()
#    return (x - mini)/(maxi-mini)
#  return(minmax)
#
num_c = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
feature_columns = []
for header in num_c:
    scal_input_fn = get_scal(header)
    feature_columns.append(feature_column.numeric_column(header)) 
#   feature_columns.append(feature_column.numeric_column(header, normalizer_fn=scal_input_fn))
print(feature_columns)
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = keras.Sequential([feature_layer,
        keras.layers.Dense(1)
        ])

model.compile(optimizer='adam',loss='mse',metrics=['mse'])
model.fit(train_ds, validation_data=val_ds, epochs=200)
loss, mse = model.evaluate(test_ds)
print("Mean Squared Error - Test Data", mse)

#print(model.summary())
