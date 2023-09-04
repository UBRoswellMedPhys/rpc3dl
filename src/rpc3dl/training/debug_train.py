# -*- coding: utf-8 -*-
"""
Created on Tue May 16 23:54:31 2023

@author: johna
"""
environment = 'CCR' # alternate - local


print("Beginning imports...")
import os
import sys
import random
from datetime import datetime
import tensorflow as tf
print(tf.__name__,tf.__version__)
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

if environment == 'CCR':
    import keras
    from keras import layers
elif environment == 'local':
    from tensorflow import keras
    from tensorflow.keras import layers


from _utils import process_surveys, window_level
from DataGenerator import InputGenerator_v2
from Resnet3DBuilder import Resnet3DBuilder
import pt_char_lookups

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
for device in physical_devices:
    config = tf.config.experimental.set_memory_growth(device, True)



print("Imports complete...")

start_time = datetime.today().strftime("%y-%m-%d_%H%M")
print("Run started at {}".format(start_time))
# ===========
# Initialize data settings

TIME_WINDOW = 'early'

if environment == 'local':
    DATA_DIR = r"E:\newdata"
    CHECKPOINT_DIR = r"D:\model_checkpoints\{}_dry_mouth\{}".format(
        TIME_WINDOW,start_time
        )
    ACTIVE_GROUPS = ['Surgery']
    
elif environment == 'CCR':
    DATA_DIR = "/projects/rpci/ahle2/johnasbach/xerostomia_outcomes/debugdata"
    CHECKPOINT_DIR = "/projects/rpci/ahle2/johnasbach/xerostomia_outcomes/debug_results/{}".format(
        start_time
        )
    ACTIVE_GROUPS = sys.argv[1:]

IPSI_CONTRA = True
AUGMENTS = True



BATCH_SIZE = 5

"""
PT_CHAR_SETTINGS = {
    'Treatment Type ' : True,
    'Age at Diagnosis ' : True,
    'Gender' : True,
    'T Stage Clinical ' : True,
    'N stage' : True,
    'HPV status' : True
    }
"""
active_fields = []
for group in ACTIVE_GROUPS:
    active_fields += pt_char_lookups.groups[group]
PT_CHAR_SETTINGS = {field : True for field in active_fields}

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
    call_augments=AUGMENTS
    )
gen.build_encoders()
gen.pt_char_settings.update(PT_CHAR_SETTINGS)
gen.build_splits(42,val=0.1,test=0.1)
gen.batch_size = BATCH_SIZE
print("Loading validation data...")
valX, valY = gen.load_all('val')
if isinstance(valX, tuple):
    valX = list(valX)

if environment == "local":
    train_dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=gen.output_sig()
        )
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_on = train_dataset
elif environment == 'CCR':
    def genwrap(gen,batch_size):
        while True:
            X = []
            Xnonvol = []
            Y = []
            while len(X) < batch_size:
                x,y = next(gen())
                if isinstance(x,tuple):
                    X.append(x[0])
                    Xnonvol.append(x[1])
                elif isinstance(x,np.ndarray):
                    X.append(x)
                Y.append(y)
            if len(Xnonvol) > 0:
                inp = [np.stack(X,axis=0),np.stack(Xnonvol,axis=0)]
            else:
                inp = np.stack(X,axis=0)
            label = np.array(Y)[:,np.newaxis]
            yield inp, label
    
    train_on = genwrap(gen,BATCH_SIZE)
        
    """
    types, shapes = gen.output_types_and_shapes()
    train_dataset = tf.data.Dataset.from_generator(
        gen,
        output_types=types,
        output_shapes=shapes
        )
    train_dataset = train_dataset.batch(BATCH_SIZE)
    """



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

if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)
    
with open(os.path.join(CHECKPOINT_DIR,"notes.txt"),"w") as f:
    f.write(notes)

gen.export_config(os.path.join(CHECKPOINT_DIR,'datagen_config.json'))

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
        train_on,
        validation_data=(valX, valY),
        # batch_size=12,
        steps_per_epoch=(len(gen.train) // BATCH_SIZE),
        epochs=400,
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