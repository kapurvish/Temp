# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 21:18:45 2019

@author: Vishal Kapur
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import numpy as np
import os
import time

tf.random.set_seed(1)

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))

char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

BATCH_SIZE = 64

BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

vocab_size = len(vocab)
embedding_dim=256
rnn_units = 1024

def build_model(vocab_size,embedding_dim,rnn_units,batch_size):
    model =tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,embedding_dim,batch_input_shape=[batch_size,None]),
    tf.keras.layers.LSTM(rnn_units,return_sequences=True,stateful=True,
                         recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
                 ])
    return model
model = build_model(
         vocab_size = len(vocab),
         embedding_dim=embedding_dim,
         rnn_units=rnn_units,
         batch_size=BATCH_SIZE
         )
print(model.summary())

for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
  
def loss(labels,logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels,logits,from_logits=True)
#eample_
  
model.compile(optimizer='adam',loss=loss)
EPOCHS=10
history=model.fit(dataset,epochs=EPOCHS)



