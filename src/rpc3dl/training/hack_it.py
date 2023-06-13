# -*- coding: utf-8 -*-
"""
Created on Tue May 16 23:54:31 2023

@author: johna
"""

import os
import random
import tensorflow as tf
import h5py
import numpy as np
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers

import build_model as models
from _utils import process_surveys

DATA_DIR = r"E:\alldata_anon"
POS_SOURCE = r"D:\alldata_anon\early_sticky_saliva_nosurgery_positive.txt"
NEG_SOURCE = r"D:\alldata_anon\early_sticky_saliva_nosurgery_negative.txt"
CHECKPOINT_DIR = r"D:\model_checkpoints\early_sticky_saliva\RUN7"

TIME_WINDOW = 'early' # 480 patients

# train patients = 290
# test patients = 46
TEST_PTS = 46
BATCH_SIZE = 10

PT_CHARS = False
INCLUDE_AUGMENTS = False

# =============================
# PUT ANY NOTES HERE

notes = """Secondary dropout decrease seemed to help a little, but I'm seeing
some heavy fluctuation of both train loss and val loss. Gut instinct here is
that learning rate is too large? It's usually hitting the comfy 0.69 range
within one or two epochs but then bounces around. The other variable I'd like
to play with is batch size.

For this run let's set up a learning rate schedule that will give it 5 epochs
of fast learning rate (0.001) then slow it down to 0.0001 for 10 and then
perhaps one more step down to 0.00005?

Also it's currently learning to be too heavy-handed on positive recs...maybe
try a slight class weight to encourage it to learn negatives?
"""




# =============================

# == NOTE ==
# using survey process now. survey fields index:
    # 2 - dry mouth
    # 3 - sticky saliva
lblidx = 3


# Build generator
def datagen(root,filelist,labeltype="early",with_chars=False):
    i = 0
    while True:
        file, key = filelist[i]
        filepath = os.path.join(root,file)
        with h5py.File(filepath,"r") as f:
            X = f[key][...]
            Y = process_surveys(f,labeltype,'mean',scale4thresh=2.7)[lblidx]
            if with_chars:
                XX = f['pt_chars'][...]
        if with_chars:
            X_ret = (X,XX)
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
with open(POS_SOURCE,"r") as f:
    pos_files = f.read()
    pos_files = pos_files.split("\n")
    
with open(NEG_SOURCE,"r") as f:
    neg_files = f.read()
    neg_files = neg_files.split("\n")
    
random.seed(98) # prev was 42
random.shuffle(pos_files)
random.shuffle(neg_files)

splitval = int(TEST_PTS/2)
train_files = pos_files[splitval:] + neg_files[splitval:]
test_files = pos_files[:splitval] + neg_files[:splitval]

print("Number of training patients:",len(train_files))
print("Number of test patients:", len(test_files))

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
random.shuffle(gen_input)

if PT_CHARS:
    outsig = ((tf.TensorSpec(shape=(40,128,128,3),dtype=tf.float32),
              tf.TensorSpec(shape=(69),dtype=tf.float32)),
              tf.TensorSpec(shape=(),dtype=tf.int32))
else:
    outsig = (tf.TensorSpec(shape=(40,128,128,3),dtype=tf.float32),
              tf.TensorSpec(shape=(),dtype=tf.int32))

train_generator = datagen(DATA_DIR, gen_input, with_chars=PT_CHARS)
train_dataset = tf.data.Dataset.from_generator(
    genwrapper(train_generator),
    output_signature=outsig
    )
# train_dataset = train_dataset.repeat(400)
train_dataset = train_dataset.batch(BATCH_SIZE)


val_volumeX = []
val_charsX = []
valY = []
for file in test_files:
    with h5py.File(os.path.join(DATA_DIR,file),"r") as f:
        val_volumeX.append(f['base'][...])
        val_charsX.append(f['pt_chars'][...])
        valY.append(
            process_surveys(f,TIME_WINDOW,'mean',scale4thresh=2.7)[lblidx]
            )
        
val_volumeX = np.array(val_volumeX)
val_charsX = np.array(val_charsX)
valY = np.array(valY)

if PT_CHARS:
    valX = [val_volumeX, val_charsX]
else:
    valX = val_volumeX

# setup callbacks, model config
checkpoint = keras.callbacks.ModelCheckpoint(
    os.path.join(CHECKPOINT_DIR,"model.{epoch:02d}-{val_loss:.2f}.h5"),
    monitor='val_auc',
    save_weights_only=False,
    save_best_only=True
    )
earlystopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=30
    )

def scheduler(epoch,lr):
    if epoch < 10:
        return 0.001
    elif 10 < epoch < 30:
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

# Build model.
with tf.device("/gpu:0"):
    model = models.single_resnet(
        input_shape=(40,128,128,3),
        with_chars=PT_CHARS,
        nonvol=69,
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
        steps_per_epoch=int(len(gen_input)/BATCH_SIZE),
        epochs=400,
        verbose=1,
        callbacks=[checkpoint, earlystopping, lrschedule],
        )


with open(os.path.join(CHECKPOINT_DIR,"train_patients.txt"),"w") as f:
    f.write("\n".join(train_files))
with open(os.path.join(CHECKPOINT_DIR,"val_patients.txt"),"w") as f:
    f.write("\n".join(test_files))
pd.DataFrame(data=history.history).to_csv(os.path.join(CHECKPOINT_DIR,"history.csv"))