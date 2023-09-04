# -*- coding: utf-8 -*-
"""
Created on Tue May 16 23:54:31 2023

@author: johna
"""
environment = 'CCR' # options: 'local', 'CCR', 'CCR_custom'
preload = True
endpoint = 'xerostomia' #alt 'xerostomia'
IPSI_CONTRA = True
AUGMENTS = True
BATCH_SIZE = 20

# Below variables are overwritten if run on CCR
KFOLDS = 5
TESTFOLD = 0


print("Beginning imports...")
import os
if environment.startswith("CCR"):
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kfold',type=int,default=None)
    parser.add_argument('-c','--cat',nargs='*',default=[])
    
import sys
import random
from datetime import datetime
import tensorflow as tf
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tensorflow import keras
from tensorflow.keras import layers

from _utils import process_surveys, window_level
from DataGenerator import InputGenerator_v2
from Resnet3DBuilder import Resnet3DBuilder
import pt_char_lookups
print("Imports complete...")

start_time = datetime.today().strftime("%y-%m-%d_%H%M")
print("Run started at {}".format(start_time))
# ===========
# Initialize data settings

TIME_WINDOW = 'early'

if environment == 'local':
    if endpoint == 'xerostomia':
        DATA_DIR = r"E:\newdata"
        CHECKPOINT_DIR = r"D:\model_checkpoints\official\{}".format(
            start_time
            )
    elif endpoint == 'survival':
        DATA_DIR = r"E:\newdata_ptv"
        CHECKPOINT_DIR = r"D:\model_checkpoints\official_survival\{}".format(
            start_time
            )
    ACTIVE_GROUPS = []
    
elif environment == 'CCR':
    args = parser.parse_args()
    if args.kfold is not None:
        KFOLDS = 5
        TESTFOLD = int(args.kfold)
        print("Running kfolds: test fold is {}".format(TESTFOLD))
    else:
        KFOLDS = None
        TESTFOLD = None
    ACTIVE_GROUPS = args.cat
    dest_dir_name = '_'.join(ACTIVE_GROUPS).replace(" ","_")
    if dest_dir_name == '':
        dest_dir_name = 'baseline'
    if TESTFOLD is not None:
        dest_dir_name += "_fold{}".format(TESTFOLD)
    DATA_DIR = "/projects/rpci/ahle2/johnasbach/xerostomia_outcomes/newdata"
    CHECKPOINT_DIR = "/projects/rpci/ahle2/johnasbach/xerostomia_outcomes/results/{}".format(
        dest_dir_name
        )
else:
    raise Exception("Environment not recognized, accepted options are 'local','CCR'")



active_fields = []
for group in ACTIVE_GROUPS:
    active_fields += pt_char_lookups.groups[group]
PT_CHAR_SETTINGS = {field : True for field in active_fields}

print(f"Conditions set\nIpsicontra rectify: {IPSI_CONTRA}")
print(f"Using augments: {AUGMENTS}\nBatch size: {BATCH_SIZE}")
print("Non-vol pt chars included: {}".format(
    [f for f,v in PT_CHAR_SETTINGS.items() if v is True]
    ))

# =============================
# PUT ANY NOTES HERE

notes = """
Running {} group
Cycling through groups to check dependencies
""".format(ACTIVE_GROUPS)
# =============================
# Prepare data
print("Starting data prep...")
gen = InputGenerator_v2(
    DATA_DIR,
    time=TIME_WINDOW,
    ipsicontra=IPSI_CONTRA,
    call_augments=AUGMENTS,
    endpoint=endpoint
    )
gen.pt_char_settings.update(PT_CHAR_SETTINGS)
gen.build_encoders()

# check top of script variables for if we're doing kfolds
if KFOLDS is not None:
    gen.build_splits(42,val=0.1,test=TESTFOLD,kfolds=KFOLDS)
else:
    gen.build_splits(42,val=0.1,test=0.1)
gen.batch_size = BATCH_SIZE
if preload is True:
    gen.preload()

print("Loading validation data...")
valX, valY = gen.load_all('val')
if isinstance(valX,tuple):
    valX = list(valX)
print("Val data loaded...")


train_dataset = tf.data.Dataset.from_generator(
    gen,
    output_signature=gen.output_sig
    )
train_dataset = train_dataset.batch(BATCH_SIZE,drop_remainder=True)



# =============================
# Setup callbacks, model config
checkpoint = keras.callbacks.ModelCheckpoint(
    os.path.join(CHECKPOINT_DIR,"model.{epoch:02d}-loss_{val_loss:.2f}-auc_{val_auc:.2f}.h5"),
    monitor='val_loss',
    save_weights_only=False,
    save_best_only=True
    )
earlystopping = keras.callbacks.EarlyStopping(
    monitor='val_auc',
    min_delta=0,
    patience=75
    )

def scheduler(epoch,lr):
    if epoch < 20:
        return 0.001
    elif 20 < epoch < 50:
        return 0.0005
    else:
        return 0.00001

lrschedule = keras.callbacks.LearningRateScheduler(
    scheduler
    )

# =========================
# Ensure write-to directory exists, this is required for model checkpointing
if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)
    
with open(os.path.join(CHECKPOINT_DIR,"notes.txt"),"w") as f:
    f.write(notes)

gen.export_config(os.path.join(CHECKPOINT_DIR,'datagen_config.json'))

# ============================
# Build model.

if gen.pt_char_len != 0:
    fusion_plan = {'late':gen.pt_char_len}
else:
    fusion_plan = {}

print("Everything's ready, building model and beginning training...")
with tf.device("/gpu:0"):
    # =================
    # Build the model
    if endpoint == 'xerostomia':
        inputshape = (40,128,128,3)
    elif endpoint == 'survival':
        inputshape = (60,128,128,3)
    model = Resnet3DBuilder.build_resnet_34(
        inputshape,
        num_outputs=1,
        fusions=fusion_plan,
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
    # =================
    # Configure run settings (fit args)
    fitargs = {
        'x':train_dataset,
        'validation_data':(valX, valY),
        'steps_per_epoch':(len(gen.train) // BATCH_SIZE),
        'epochs':400,
        'callbacks':[checkpoint, earlystopping, lrschedule],
        }
    if environment == 'local':
        fitargs['verbose'] = 1
    elif environment == 'CCR':
        fitargs['verbose'] = 2
        
    history = model.fit(**fitargs)

# save the specific patient files used for train, val, test
for group in ['train','val','test']:
    with open(os.path.join(CHECKPOINT_DIR,f"{group}_patients.txt"),"w") as f:
        f.write("\n".join(getattr(gen,group)))

#
pd.DataFrame(data=history.history).to_csv(os.path.join(CHECKPOINT_DIR,"history.csv"))    

# ===============
# free up some memory space
del valX
del valY

with open(os.path.join(CHECKPOINT_DIR,"test_patients.txt"),"w") as f:
    f.write("\n".join(gen.test))

testX, testY = gen.load_all('test')
if isinstance(testX,tuple):
    testX = list(testX)
preds = model.predict(testX)

results = {
    'patients' : gen.test,
    'true' : np.squeeze(testY),
    'preds' : np.squeeze(preds)
    }
pd.DataFrame(data=results).to_csv(os.path.join(CHECKPOINT_DIR,"results.csv"))

print('AUC:',roc_auc_score(np.squeeze(testY),np.squeeze(preds)))