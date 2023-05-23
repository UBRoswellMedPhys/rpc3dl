# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 19:09:54 2022

@author: johna
"""

from tensorflow import keras
from tensorflow.keras import layers

def get_model(config):
    """
    Guide function which reads the config settings and calls the appropriate
    model architecture for what is requested.
    """
    
    input_shape = config.get('data_settings','volume_shape')
    input_shape = input_shape.split(",")
    input_shape = [int(dim) for dim in input_shape]
    
    if config.getboolean('data_settings','withmask',fallback=True) is True:
        channels = 3
    else:
        channels = 2
        
    input_shape.append(channels)
    # base_filters = config.getint('model_settings','base_filters')
    
    with_chars = config.getboolean('data_settings','patient_chars')
    
    if config.getboolean('data_settings','single') is True:
        model = single_resnet(input_shape=input_shape,
                              with_chars=with_chars,
                              nonvol=38)
    else:
        model = dual_resnet(input_shape=input_shape,
                            with_chars=with_chars,
                            nonvol=38)
            
    return model




def res_block(input_shape,
              filters=32):
    
    inputs = keras.Input(input_shape)
    ## First layer
    conv1 = layers.Conv3D(
        filters=filters, kernel_size=(5, 5, 5),strides=(1,2,2), 
        padding="same", activation="relu"
        )(inputs)
    conv11 = layers.Conv3D(
        filters=filters, kernel_size=(5, 5, 5), strides=(1,1,1), 
        padding="same"
        )(conv1)
    norm1 = layers.BatchNormalization(axis=-1)(conv11)
    relu1 = layers.Activation("relu")(norm1)
    residual1 = layers.Conv3D(
        filters=filters, kernel_size=(3, 3, 3), strides=(1,1,1), 
        padding="same", activation="relu"
        )(relu1)
    resblock1 = layers.Add()([conv1, residual1])
    
    completeblock = keras.Model(inputs=inputs,outputs=resblock1)
    return completeblock

def organ_path(input_shape=(40,96,96,3),filter_depth=32):
    inputs = keras.Input(input_shape)
    res1 = res_block(input_shape,filters=filter_depth)(inputs)
    res2 = res_block(res1.shape[1:],filters=filter_depth)(res1)
    res3 = res_block(res2.shape[1:],filters=filter_depth)(res2)
    organ_model = keras.Model(inputs=inputs,outputs=res3)
    return organ_model

def non_volume_path(length=38):
    inputs = keras.Input(shape=(length,))
    x = layers.Dense(units=32, activation="relu")(inputs)
    x = layers.Dense(units=32, activation="relu")(x)
    model = keras.Model(inputs=inputs,outputs=x)
    return model
    
def dual_resnet(input_shape=(40,96,96,3),
                with_chars=True,
                nonvol=38):
    input1 = keras.Input(input_shape)
    input2 = keras.Input(input_shape)
    left_parotid = organ_path(input_shape)(input1)
    right_parotid = organ_path(input_shape)(input2)
    merged = layers.Concatenate()([left_parotid,right_parotid])
    merged = layers.Conv3D(
        filters=64,kernel_size=(5,3,3),strides=(1,1,1),
        padding="valid",activation="relu"
        )(merged)
    pooled = layers.GlobalAveragePooling3D()(merged)
    x = layers.Dense(units=32,activation="relu")(pooled)
    
    if with_chars:
        nonvolinput = keras.Input(shape=(nonvol,))
        y = non_volume_path(length=nonvol)(nonvolinput)
        x = layers.Concatenate()([x,y])
        
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    final = layers.Dense(1,activation='sigmoid')(x)
    
    if with_chars:
        allinputs = [input1,input2,nonvolinput]
    else:
        allinputs = [input1,input2]
    
    model = keras.Model(inputs=allinputs,outputs=final)
    return model

def single_resnet(input_shape=(40,256,256,3),
                  organ_resnet_depth=32,
                  with_chars=True,
                  nonvol=38):
    inputs = keras.Input(input_shape)
    x = organ_path(input_shape,filter_depth=organ_resnet_depth)(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv3D(
        filters=32, kernel_size=(5, 5, 5), strides=(1, 2, 2),
        padding='valid', activation='relu'
        )(x)
    x = layers.Conv3D(
        filters=32, kernel_size=(3,3,3), strides=(2, 2, 2),
        padding='valid',activation='relu'
        )(x)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=16,activation="relu")(x)
    if with_chars:
        nonvolinput = keras.Input(shape=(nonvol,))
        y = non_volume_path(length=nonvol)(nonvolinput)
        x = layers.Concatenate()([x,y])
    x = layers.Dense(units=16,activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=8,activation="relu")(x)
    final = layers.Dense(units=1,activation="sigmoid")(x)
    
    if with_chars:
        allinputs = [inputs,nonvolinput]
    else:
        allinputs = inputs
    
    model = keras.Model(inputs=allinputs,outputs=final)
    return model
