# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 19:09:54 2022

@author: johna
"""

from tensorflow import keras
from tensorflow.keras import layers

def get_model(width=90, height=60, depth=40,channels=2):
    """Build a 3D convolutional neural network model."""

    l_inputs = keras.Input((depth, width, height, channels))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(l_inputs)
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    


    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    
    r_inputs = keras.Input((depth, width, height, channels))

    xx = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(r_inputs)
    xx = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(xx)
    xx = layers.MaxPool3D(pool_size=2)(xx)
    xx = layers.BatchNormalization()(xx)

    xx = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(xx)
    xx = layers.MaxPool3D(pool_size=2)(xx)
    xx = layers.BatchNormalization()(xx)
    
    x = layers.Concatenate()([x,xx])

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model((l_inputs,r_inputs), outputs)
    return model

def unified_model(width=256,height=256,depth=40,channels=3):
    
    inputs = keras.Input((depth,width,height,channels))
    
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    


    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)
    
    model = keras.Model(inputs, outputs)
    return model