import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers 
from tensorflow.keras.applications.densenet import DenseNet121

def conv_3D(input_shape, num_classes):
    model = tf.keras.models.Sequential([
      
      layers.Conv3D(128, (3,3,3), padding='same', input_shape=input_shape),
      layers.BatchNormalization(),
      layers.Activation('relu'),
      layers.Conv3D(128, (3,3,3), padding='same', input_shape=input_shape),
      layers.BatchNormalization(),
      layers.Activation('relu'),

      layers.MaxPool3D((2,2,2), strides=(2,2,2)),

      layers.Conv3D(64, (3,3,3), padding='same'),
      layers.BatchNormalization(),
      layers.Activation('relu'),
      layers.Conv3D(64, (3,3,3), padding='same'),
      layers.BatchNormalization(),
      layers.Activation('relu'),

      layers.MaxPool3D((2,2,2), strides=(2,2,2)),

      layers.Conv3D(32, (3,3,3), padding='same'),
      layers.BatchNormalization(),
      layers.Activation('relu'),
      layers.Conv3D(32, (3,3,3), padding='same'),
      layers.BatchNormalization(),
      layers.Activation('relu'),

      layers.MaxPool3D((2,2,2), strides=2),

      layers.Flatten(),

      layers.Dense(512, activation='relu'),
      layers.Dropout(0.2),
      layers.Dense(256, activation='relu'),
      layers.Dropout(0.2),
      layers.Dense(num_classes, activation='softmax')
     
    ])
    return model

