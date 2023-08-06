# -*- coding: utf-8 -*-
"""
Created on Tue May 16 23:54:31 2023

@author: johna
"""
print("Beginning imports...")
import os
import random
import tensorflow as tf
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tensorflow import keras
from tensorflow.keras import layers

import build_model as models
from _utils import process_surveys, window_level
from DataGenerator import InputGenerator_v2
from Resnet3DBuilder import Resnet3DBuilder
print("Imports complete...")

DATA_DIR = r"E:\newdata"


TIME_WINDOW = 'early'

CHECKPOINT_DIR = r"D:\model_checkpoints\{}_dry_mouth\RUN10".format(TIME_WINDOW)

BATCH_SIZE = 20

PT_CHAR_SETTINGS = {
    'Treatment Type ' : True,
    'Age at Diagnosis ' : True,
    'Gender' : True,
    'T Stage Clinical ' : True,
    'N stage' : True,
    'HPV status' : True
    }

# =============================
# PUT ANY NOTES HERE

notes = """

Trying out adding smoking status, and changed seed

"""
# =============================
# Prepare data
print("Starting data prep...")
gen = InputGenerator_v2(DATA_DIR,time='early',ipsicontra=False)
gen.build_encoders()
gen.pt_char_settings.update(PT_CHAR_SETTINGS)
gen.build_splits(98,val=0.1,test=0.1)
gen.batch_size = BATCH_SIZE
print("Loading validation data...")
valX, valY = gen.load_all('val')

train_dataset = tf.data.Dataset.from_generator(
    gen,
    output_signature=gen.output_sig()
    )

train_dataset = train_dataset.batch(BATCH_SIZE)

# =============================
# setup callbacks, model config
checkpoint = keras.callbacks.ModelCheckpoint(
    os.path.join(CHECKPOINT_DIR,"model.{epoch:02d}-loss_{val_loss:.2f}-auc_{val_auc:.2f}.h5"),
    monitor='val_loss',
    save_weights_only=False,
    save_best_only=True
    )
earlystopping = keras.callbacks.EarlyStopping(
    monitor='val_auc',
    min_delta=0,
    patience=32
    )

def scheduler(epoch,lr):
    if epoch < 20:
        return 0.001
    elif 20 < epoch < 40:
        return 0.0005
    else:
        return 0.00001

lrschedule = keras.callbacks.LearningRateScheduler(
    scheduler
    )

if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)
    
with open(os.path.join(CHECKPOINT_DIR,"notes.txt"),"w") as f:
    f.write(notes)

# ============================
# Build model.
with tf.device("/gpu:0"):
    model = Resnet3DBuilder.build_resnet_34(
        (40,128,128,3),
        num_outputs=1,
        fusions={'late':gen.pt_char_len},
        basefilters=32
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
        steps_per_epoch=(len(gen.train) // BATCH_SIZE),
        epochs=200,
        verbose=1,
        callbacks=[checkpoint, earlystopping, lrschedule],
        )


with open(os.path.join(CHECKPOINT_DIR,"train_patients.txt"),"w") as f:
    f.write("\n".join(gen.train))
with open(os.path.join(CHECKPOINT_DIR,"val_patients.txt"),"w") as f:
    f.write("\n".join(gen.val))

pd.DataFrame(data=history.history).to_csv(os.path.join(CHECKPOINT_DIR,"history.csv"))    

# free up some memory space
del valX
del valY

with open(os.path.join(CHECKPOINT_DIR,"test_patients.txt"),"w") as f:
    f.write("\n".join(gen.test))

testX, testY = gen.load_all('test')
preds = model.predict(testX)

results = {
    'patients' : gen.test,
    'true' : np.squeeze(testY),
    'preds' : np.squeeze(preds)
    }
pd.DataFrame(data=results).to_csv(os.path.join(CHECKPOINT_DIR,"results.csv"))

print('AUC:',roc_auc_score(np.squeeze(testY),np.squeeze(preds)))