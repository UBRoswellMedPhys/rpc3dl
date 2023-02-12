# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 22:11:57 2022

@author: johna
"""

import os
import configparser
import math
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import _utils as util
from build_model import get_model

def IDE_config():
    
    LEARNRATE = 0.001
    WITH_MASK = True
    SINGLE = True
    BATCH_SIZE = 3
    CLASS_BAL = 'oversample'
    EPOCHS = 80
    BASE_FILTERS = 16
    
    config = configparser.ConfigParser()
    config['data_settings'] = {
        'normalize':True,
        'withmask':WITH_MASK,
        'masked':False,
        'wl':True,
        'wl_window':400,
        'wl_level':50,
        'dose_norm':False,
        'ipsi_contra':False,
        'single':SINGLE,
        'class_balance':CLASS_BAL,
        'augment':False,
        'batch_size':BATCH_SIZE,
        'patient_chars':True,
        'shuffle':True,
        'anonymized':False
        }
    
    config['model_settings'] = {
        'base_filters': BASE_FILTERS,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'initial_learnrate': LEARNRATE
        }
    
    config['filepaths'] = {
        'source': "/home/johnasbach/Research/arrays",
        'labelfile': "/home/johnasbach/Research/resources/early_dry_mouth_label.csv",
        'artifacts': "/home/johnasbach/Research/training_runs",
        'ohe_pt_chars': "/home/johnasbach/Research/resources/ohe_patient_char.csv"
        }
    
    return config

def run(config):
    print("Beginning model training process")

    SOURCE_FOLDER = config.get('filepaths', 'source')
    ANON = config.getboolean('data_settings','anonymized')

    if ANON:
        idcol = 'ANON_ID'
    elif not ANON:
        idcol = 'MRN'    
    
    # === Prep one-hot encoded patient characteristics ===
    onehotptchars = pd.read_csv(config.get('filepaths','ohe_pt_chars'))
    onehotptchars = onehotptchars.set_index(idcol)
    onehotptchars.replace(to_replace="90+",value=90,inplace=True)
    onehotptchars['age'] = onehotptchars['age'].astype(int)
    
    labels = pd.read_csv(config['filepaths']['labelfile'])
    labels = labels.set_index('MRN')
    all_labels = labels.to_dict()['label']
                
    all_oh = onehotptchars.dropna() # only want to save patients that we have all data
    
    print("Patient characteristics and labels successfully loaded.")
    
    exclude = ['428612']
    # no need to review 428612 at this time, it has an empty mask
    
    all_pts = [
        int(pt) for pt in os.listdir(SOURCE_FOLDER) if all((
            os.path.isdir(os.path.join(SOURCE_FOLDER,pt)),
            pt not in exclude,
            int(pt) in all_labels.keys()
            ))
        ]
    
    pos_pt = [pt for pt in all_pts if all_labels[pt]==1]
    neg_pt = [pt for pt in all_pts if all_labels[pt]==0]
    np.random.shuffle(pos_pt)
    np.random.shuffle(neg_pt)
    
    print("Patients split into pos and neg, {} and {}, respectively".format(
        len(pos_pt),
        len(neg_pt)
    ))

    val_pts = list(pos_pt[:20]) + list(neg_pt[:10])
    print("Validation patient IDs:",val_pts)
    train_pts = [pt for pt in all_pts if pt not in val_pts]
    
    
    train_labels = {pt:all_labels[pt] for pt in train_pts}
    val_labels = {pt:all_labels[pt] for pt in val_pts}
    
    if config.get('data_settings', 'class_balance') != "off":
        num_pos = sum(train_labels.values())
        num_neg = len(train_labels) - num_pos
        small_class = min((num_pos,num_neg))
        if config.get('data_settings', 'class_balance') == 'undersample':
            dataset_size = small_class * 2
        elif config.get('data_settings', 'class_balance') == 'oversample':
            numtrain = len(train_labels)
            shave = numtrain % config.getint('model_settings','batch_size')
            dataset_size = numtrain - shave
    
    else:
        dataset_size = len(train_labels)
    
    if config.getboolean('data_settings', 'patient_chars') is True:
        ptchars = all_oh
    elif config.getboolean('data_settings', 'patient_chars') is False:
        ptchars = None
    
    
    inputdata = util.gen_inputs(
        config=config,
        labels=train_labels,
        ptchars=ptchars,
        training=True
        )

    valinputdata = util.gen_inputs(
        config=config,
        labels=val_labels,
        ptchars=ptchars,
        training=False
        )
    
    if config.getboolean('data_settings','single') is True:
        if config.getboolean('data_settings','patient_chars') is True:
            num_inputs = 2
        else:
            num_inputs = 1
    else:
        if config.getboolean('data_settings','patient_chars') is True:
            num_inputs = 3
        else:
            num_inputs = 2
    
    batch_input = util.batcher(
        inputdata,
        batch_size=config.getint('model_settings','batch_size'),
        num_inputs=num_inputs
        )
    val_batch = util.batcher(
        valinputdata,
        batch_size=len(val_pts),
        num_inputs=num_inputs
        )
    
    print("Loading validation data in whole")
    valX, valY = next(val_batch) # generates entire validation dataset
    
    # is this necessary? should it be modifiable?
    def lr_schedule(epoch,lr):
        if epoch < 10:
            return lr
        elif 10 <= epoch < 20:
            return lr*0.9
        else:
            return max(lr*0.5, 0.00001)
        
    callback1 = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    callback2 = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
        )
    
    if config.getboolean('data_settings','withmask',fallback=True):
        channels = 3
    else:
        channels = 2
    
    print("Building model")
    model = get_model(config)
    
    print("Compiling model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            lr=config.getfloat('model_settings', 'initial_learnrate')
            ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(),
                 tf.keras.metrics.FalseNegatives(),
                 tf.keras.metrics.FalsePositives()]
        )
    #model.summary()

    history = model.fit(
        x=batch_input,
        steps_per_epoch=math.floor(
            dataset_size/config.getint('model_settings','batch_size')
            ),
        validation_data=(valX,valY),
        epochs=config.getint('model_settings','epochs'),
        verbose=1,
        callbacks=[callback1,callback2])
    
    # once complete, save run details
    destination = config.get('filepaths','artifacts')
    prev_runs = [x for x in os.listdir(destination) if x.startswith("RUN")]
    run_nums = [int(x.split("_")[-1]) for x in prev_runs]
    latest_run = max(run_nums)
    new_run = latest_run + 1
    new_run_dest = os.path.join(destination,"RUN_{}".format(new_run))
    os.mkdir(new_run_dest)
    
    details = pd.DataFrame(data=history.history)
    details.to_csv(os.path.join(new_run_dest,"history.csv"),index=False)
    
    with open(os.path.join(new_run_dest,"config.ini"),"w") as configwritefile:
        config.write(configwritefile)
        configwritefile.close()
    
    validation_df = pd.DataFrame(
        index=list(val_labels.keys()),
        columns=["True","Predicted"]
        )
    predicted = model.predict(valX)
    validation_df['True'] = np.squeeze(valY)
    validation_df['Predicted'] = np.squeeze(predicted)
    validation_df.index = validation_df.index.rename("MRN")
    validation_df.to_csv(os.path.join(new_run_dest,"validation_results.csv"))
        
    tf.keras.utils.plot_model(
        model,
        to_file=os.path.join(new_run_dest,"model_plot.png"),
        show_shapes=True
        )

if __name__ == "__main__":
    import argparse
    IDE = False
    
    if IDE is False:
        cli = argparse.ArgumentParser()
        cli.add_argument("configpath")
        clargs = cli.parse_args()
        config = configparser.ConfigParser()
        config.read(clargs.configpath)
    elif IDE is True:
        config = IDE_config()
    
    run(config)