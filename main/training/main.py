# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 22:11:57 2022

@author: johna
"""

import os
import math
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import _utils as util
from build_model import get_model, unified_model, get_model_w_ptchars

SOURCE_FOLDER = r"D:\extracteddata"

LEARNRATE = 0.001
WITH_MASK = True
SINGLE = False
BATCH_SIZE = 10
CLASS_BAL = 'oversample'
EPOCHS = 80
BASE_FILTERS = 16

settings = {
    'epochs':EPOCHS+1,
    'normalize':True,
    'withmask':WITH_MASK,
    'masked':False,
    'wl':True,
    'dose_norm':True,
    'ipsi_contra':True,
    'single':SINGLE,
    'class_balance':CLASS_BAL,
    'augment':True,
    'batch_size':BATCH_SIZE,
    'patient_chars':True
    }

model_settings = {
    'base_filters': BASE_FILTERS,
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE
    }

# === Prep one-hot encoded patient characteristics ===
onehotptchars = pd.read_csv("D:\H_N\ohe_patient_char.csv")
onehotptchars = onehotptchars.set_index("ANON_ID")
onehotptchars.replace(to_replace="90+",value=90,inplace=True)
onehotptchars['age'] = onehotptchars['age'].astype(int)

labels = pd.read_csv(r"D:\extracteddata\newlabels.csv")
# === Build mapper ===
labelmap = {}
for i,row in labels.iterrows():
    if row['OLDID'] not in labelmap.keys():
        labelmap[row['OLDID']] = row['ID']
        
old_id_oh = pd.DataFrame(index=labelmap.keys(),
                         columns=onehotptchars.columns)
for k,v in labelmap.items():
    if v in list(onehotptchars.index):
        old_id_oh.at[k,:] = onehotptchars.loc[v]
    
all_oh = onehotptchars.append(old_id_oh)
all_oh = all_oh.dropna()

oldid_labels = pd.DataFrame(data={'ID':labels['OLDID'].values,
                                  'label':labels['label'].values})
oldid_labels = oldid_labels.dropna()
oldid_labels = oldid_labels.set_index('ID')
oldid_labels = oldid_labels.to_dict()['label']

newid_labels = pd.DataFrame(data={'ID':labels['ID'].values,
                                  'label':labels['label'].values})
newid_labels = newid_labels.dropna()
newid_labels = newid_labels.set_index('ID')
newid_labels = newid_labels.to_dict()['label']

all_labels = {}
for d in [oldid_labels,newid_labels]:
    for k,v in d.items():
        all_labels[k] = v

exclude = ['018_019','018_033','018_038','018_056','018_069','018_089',
          '018_091','018_102','018_117','018_120',
           '018_126','018_128','018_130','018_131',
           'ANON_016','ANON_023','ANON_027']

exclude += ['018_011'] # dose data is wrong

# 018_019 has different shaped dose beams, dose data is wrong
# 018_033 is missing dose files, dose data is wrong
# 018_038 has no label data
# 018_056 dose expand error
# 018_069 no label data
# 018_089 has different slice thicknesses, dose data is no good

# 018_091 has a dose expand error, needs review
# 018_102 - unknown, needs review
# 018_117 - unknown
# 018_120 - unknown
# 018_126 - unknown
# 018_128 - unknown
# 018_130 - unknown
# 018_131 - unknown
# ANON_016 - expand dose doesn't work
# ANON_023 no label data (need to double check MRN mappings)
# ANON_027 same as above


all_pts = [
    pt for pt in os.listdir(SOURCE_FOLDER) if all((
        os.path.isdir(os.path.join(SOURCE_FOLDER,pt)),
        pt not in exclude,
        pt in all_labels.keys()
        ))
    ]

pos_pt = [pt for pt in all_pts if all_labels[pt]==1]
neg_pt = [pt for pt in all_pts if all_labels[pt]==0]
np.random.shuffle(pos_pt)
np.random.shuffle(neg_pt)

val_pts = list(pos_pt[:10]) + list(neg_pt[:8])
print("Validation patient IDs:",val_pts)
train_pts = [pt for pt in all_pts if pt not in val_pts]


train_labels = {pt:all_labels[pt] for pt in train_pts}
val_labels = {pt:all_labels[pt] for pt in val_pts}

if CLASS_BAL:
    num_pos = sum(train_labels.values())
    num_neg = len(train_labels) - num_pos
    small_class = min((num_pos,num_neg))
    if CLASS_BAL == 'undersample':
        dataset_size = small_class * 2
    elif CLASS_BAL == 'oversample':
        dataset_size = len(train_labels) - (len(train_labels) % BATCH_SIZE)

else:
    dataset_size = len(train_labels)

if settings['patient_chars'] is True:
    ptchars = all_oh
elif settings['patient_chars'] is False:
    ptchars = None


inputdata = util.gen_inputs(
    source_dir=SOURCE_FOLDER,
    labels=train_labels,
    ptchars=ptchars,
    **settings
    )

valinputdata = util.gen_inputs(
    source_dir=SOURCE_FOLDER,
    labels=val_labels,
    ptchars=ptchars,
    epochs=1,
    normalize=settings['normalize'],
    withmask=settings['withmask'],
    masked=settings['masked'],
    wl=settings['wl'],
    dose_norm=settings['dose_norm'],
    ipsi_contra=settings['ipsi_contra'],
    single=settings['single'],
    class_balance=False
    )


batch_input = util.batcher(inputdata,batch_size=BATCH_SIZE)
val_batch = util.batcher(valinputdata,batch_size=len(val_pts))

val_data = next(val_batch)

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

if SINGLE is False and settings['patient_chars'] is False:
    model = get_model(
        channels=channels,
        base_filters=model_settings['base_filters']
        )
elif SINGLE is False and settings['patient_chars'] is True:
    model = get_model_w_ptchars(
        channels=channels,
        base_filters=model_settings['base_filters']
        )
elif SINGLE is True:
    model = unified_model(
        channels=channels,
        base_filters=model_settings['base_filters']
        )
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNRATE),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.FalseNegatives(),
                       tf.keras.metrics.FalsePositives()])
print(model.summary())
history = model.fit(x=batch_input,
                    steps_per_epoch=math.floor(dataset_size/BATCH_SIZE),
                    validation_data=val_data,
                    epochs=EPOCHS,
                    verbose=1,
                    callbacks=[callback])

# once complete, save run details
destination = r"D:\nn_training_results"
prev_runs = [x for x in os.listdir(destination) if x.startswith("RUN")]
run_nums = [int(x.split("_")[-1]) for x in prev_runs]
latest_run = max(run_nums)
new_run = latest_run + 1
new_run_dest = os.path.join(destination,"RUN_{}".format(new_run))
os.mkdir(new_run_dest)

details = pd.DataFrame(data=history.history)
details.to_csv(os.path.join(new_run_dest,"history.csv"),index=False)
all_settings = {**settings, **model_settings}
with open(os.path.join(new_run_dest,"data_settings.json"),"w+") as f:
    json.dump(all_settings,f)
    f.close()
with open(os.path.join(new_run_dest,"val_patients.txt"),"w+") as f:
    for pt in val_pts:
        f.write(pt + "\n")
    f.close()
    
tf.keras.utils.plot_model(
    model,
    to_file=os.path.join(new_run_dest,"model_plot.png"),
    show_shapes=True
    )



# testresults = pd.DataFrame(columns=['ID','label','prediction'])
# for k,v in test_labels.items():
#     test_inputdata = util.gen_inputs(
#         {k:v},
#         epochs=1,
#         normalize=True,
#         withmask=WITH_MASK,
#         masked=False,
#         wl=True,
#         dose_norm=True,
#         ipsi_contra=True,
#         single=SINGLE,
#         shuffle=False
#         )
#     X, Y = next(test_inputdata)
#     if SINGLE is False:
#         X = (X[0][np.newaxis,...],X[1][np.newaxis,...])
#     elif SINGLE is True:
#         X = X[np.newaxis,...]
#     pred = np.squeeze(model.predict(X))
#     testresults = testresults.append(
#         {'ID':k,'label':v,'prediction':pred},
#         ignore_index=True
#         )