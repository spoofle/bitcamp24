# -*- coding: utf-8 -*-
"""simple-lstm.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/11IUvlIDr2K12w00P3-g0-Ichm2gfxvFi
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

sequence_length = 10 # to change soon - based on data
input_features = 1 # unilateral dataset?

# random data generation in lieu of no data so far
X = np.random.randn(100, sequence_length, input_features)  # x value feature
y = np.random.randint(0, 2, size=(100,)) # y value feature

# lstm model
model = Sequential([
    LSTM(50, input_shape=(sequence_length, input_features)), # 50 neurons
    Dense(1, activation='sigmoid') # dense layer with 1 unit
])

# model compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# adam - minimize the loss function during the training of neural networks
# binary_crossentropy - binary classification
# accuracy - evaluation metric

# model training
model.fit(X, y, epochs=10, batch_size=32)
# epochs as 10 - 10 iterations over the dataset
# batch size as 32 - 32 samples per epoch

loss, accuracy = model.evaluate(X, y) # calculate loss + accuracy
print(f'Loss: {loss}, Accuracy: {accuracy}')