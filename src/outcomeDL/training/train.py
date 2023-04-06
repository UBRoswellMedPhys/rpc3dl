# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 22:42:31 2023

@author: johna
"""

import os
import configparser
import tensorflow as tf
from tensorflow import keras

import generator
import build_model

"""
Ultimately we'll have a config file that is divided into multiple sections:
    Data preprocessing
    Model definition
    Training process
    Evaluation?
"""

config = configparser.ConfigParser()
config.read("path/to/config")

BATCH_SIZE = 10
EPOCHS = 200
CALLBACKS = []
# TODO - set up callbacks. For sure want ModelCheckpoint. Possibly CSVLogger.
# Also considering: LearningRateScheduler or ReduceLROnPlateau

ROOT_DATA_DIR = 'path/to/dir'
# principle: clean/filter data prior to training so that this script can
# rely on everything within ROOT_DATA_DIR

idlist = []
for e in os.listdir(ROOT_DATA_DIR):
    fullpath = os.path.join(ROOT_DATA_DIR,e)
    if os.path.isdir(fullpath):
        idlist.append(e)
   
val_idlist = [] #placeholder
"""
==========================
TODO
Insert logic for distinguishing validation patients here

Assumption: patient IDs that are val will be REMOVED from idlist
to form a separate list.
==========================
"""


traingen = generator.data_generator(idlist,ROOT_DATA_DIR)
valgen = generator.data_generator(val_idlist,ROOT_DATA_DIR)

train_input = tf.data.Dataset.from_generator(traingen).batch(BATCH_SIZE)
val_input = tf.data.Dataset.from_generator(valgen)



model = build_model.get_model(config)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        lr=config.getfloat('model_settings', 'initial_learnrate')
        ),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy(),
             tf.keras.metrics.FalseNegatives(),
             tf.keras.metrics.FalsePositives()]
    )

history = model.fit(
    x=train_input,
    validation_data=val_input,
    epochs=EPOCHS,
    verbose=1,
    callbacks=CALLBACKS
)

predictions = model.predict(val_input)

# TODO - Build evaluation module
# TODO - Save necessary artifacts