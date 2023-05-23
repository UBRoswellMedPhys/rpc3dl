# -*- coding: utf-8 -*-
"""
Created on Tue May 16 23:54:31 2023

@author: johna
"""

import os
import tensorflow as tf
import h5py
import numpy as np
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers

import build_model as models

DATA_DIR = r"E:\alldata\data_files"
TRAIN_SOURCE = r"D:\alldata\early_train.txt"
TEST_SOURCE = r"D:\alldata\early_test.txt"
CHECKPOINT_DIR = r"D:\model_checkpoints\early\RUN18"

BATCH_SIZE = 30

INCLUDE_AUGMENTS = True





# Build generator
def datagen(root,filelist,labeltype="early",with_chars=False):
    i = 0
    while True:
        file, key = filelist[i]
        filepath = os.path.join(root,file)
        with h5py.File(filepath,"r") as f:
            X = f[key][...]
            Y = f['labels'].attrs.get('early')
            if with_chars:
                XX = f['pt_chars'][...]
        if with_chars:
            X_ret = [X,XX]
        else:
            X_ret = X
        i += 1
        if i == len(filelist):
            i = 0
        yield X_ret, Y
        
def genwrapper(generator,batch_size=20):
    def gencall(generator=generator):
        for X,Y in generator:
            yield X,Y
    return gencall
    
# Build datasets
with open(TRAIN_SOURCE,"r") as f:
    train_files = f.read()
    train_files = train_files.split("\n")

# Check files to see which have valid augments
gen_input = [(file,"base") for file in train_files]
if INCLUDE_AUGMENTS is True:
    for file in train_files:
        filepath = os.path.join(DATA_DIR,file)
        with h5py.File(filepath,"r") as f:
            if f['augment_1'][...].shape == f['base'][...].shape:
                gen_input.append((file,"augment_1"))
            else:
                gen_input.append((file,"base"))
            if f['augment_2'][...].shape == f['base'][...].shape:
                gen_input.append((file,"augment_2"))
            else:
                gen_input.append((file,"base"))
np.random.shuffle(gen_input)

with open(TEST_SOURCE,"r") as f:
    test_files = f.read()
    test_files = test_files.split("\n")
    
    


train_generator = datagen(DATA_DIR, gen_input)
train_dataset = tf.data.Dataset.from_generator(
    genwrapper(train_generator),
    output_signature=(tf.TensorSpec(shape=(40,128,128,3),dtype=tf.float32),
                      tf.TensorSpec(shape=(),dtype=tf.int32))
    )
# train_dataset = train_dataset.repeat(400)
train_dataset = train_dataset.batch(BATCH_SIZE)


valX = []
valY = []
for file in test_files:
    with h5py.File(os.path.join(DATA_DIR,file),"r") as f:
        valX.append(f['base'][...])
        valY.append(f['labels'].attrs['early'])
        
valX = np.array(valX)
valY = np.array(valY)

# setup callbacks, model config
checkpoint = keras.callbacks.ModelCheckpoint(
    os.path.join(CHECKPOINT_DIR,"model.{epoch:02d}-{val_loss:.2f}.h5"),
    save_weights_only=False,
    save_best_only=True
    )
earlystopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=30
    )

if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

# Build model.
with tf.device("/cpu:0"):
    model = models.single_resnet(
        input_shape=(40,128,128,3),
        with_chars=False,
        organ_resnet_depth=16
        )
    optim = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optim,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.AUC(name='auc'),
                 keras.metrics.BinaryAccuracy(name='acc'),
                 keras.metrics.Precision(name='prec'),
                 keras.metrics.Recall(name='rec')],
        )
    
    

    history = model.fit(
        train_dataset,
        validation_data=(valX, valY),
        # batch_size=12,
        steps_per_epoch=int(len(gen_input) / BATCH_SIZE),
        epochs=400,
        verbose=1,
        callbacks=[checkpoint, earlystopping],
        class_weight={0:0.85,1:1.0}
        )


pd.DataFrame(data=history.history).to_csv(os.path.join(CHECKPOINT_DIR,"history.csv"))