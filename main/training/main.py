# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 22:11:57 2022

@author: johna
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import _utils as util
from build_model import get_model, unified_model

LEARNRATE = 0.001
WITH_MASK = True


labels = pd.read_csv(r"D:\extracteddata\labels.csv")
labels = labels.set_index('patient').to_dict()
labels = labels['xero']

exclude = ['018_019','018_033','018_089','018_102','018_117',
           '018_120','018_126','018_128','018_130','018_131']

# 018_019 has different shaped dose beams
# 018_033 is missing dose files
# 018_089 has different slice thicknesses
# 018_102 - unknown, needs review
# 018_117 - unknown
# 018_120 - unknown
# 018_126 - unknown
# 018_128 - unknown
# 018_130 - unknown
# 018_131 - unknown

for pt in exclude:
    del labels[pt]


inputdata = util.gen_inputs(
    labels,
    epochs=51,
    normalize=True,
    withmask=WITH_MASK,
    masked=False,
    wl=True,
    dose_norm=True,
    ipsi_contra=True,
    single=True
    )

def lr_schedule(epoch,lr):
    if epoch < 10:
        return lr
    elif 10 <= epoch < 20:
        return lr*0.9
    else:
        return max(lr*0.5, 0.00001)
    
callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

if WITH_MASK:
    channels = 3
else:
    channels = 2

#model = get_model(channels=channels)
model = unified_model(channels=channels)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNRATE),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.FalseNegatives(),
                       tf.keras.metrics.FalsePositives()])
history = model.fit(x=inputdata,steps_per_epoch=61,epochs=50,verbose=1,callbacks=[callback])