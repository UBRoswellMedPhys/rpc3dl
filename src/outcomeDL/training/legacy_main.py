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
        'augment':True,
        'batch_size':BATCH_SIZE,
        'patient_chars':True,
        'shuffle':True
        }
    
    config['model_settings'] = {
        'base_filters': BASE_FILTERS,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'initial_learnrate': LEARNRATE
        }
    
    config['filepaths'] = {
        'source': r"D:\extracteddata",
        'labelfile': r"D:\extracteddata\latelabels.csv",
        'artifacts': r"D:\nn_training_results",
        'ohe_pt_chars': r"D:\H_N\ohe_patient_char.csv"
        }
    
    return config

def run(config):
    SOURCE_FOLDER = config.get('filepaths', 'source')    
    
    # === Prep one-hot encoded patient characteristics ===
    onehotptchars = pd.read_csv(config.get('filepaths','ohe_pt_chars'))
    if "MRN" in onehotptchars.columns:
        onehotptchars = onehotptchars.set_index("MRN")
    elif "ANON_ID" in onehotptchars.columns:
        onehotptchars = onehotptchars.set_index("ANON_ID")
    onehotptchars.replace(to_replace="90+",value=90,inplace=True)
    onehotptchars['age'] = onehotptchars['age'].astype(int)
    
    labels = pd.read_csv(config['filepaths']['labelfile'])
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
    all_oh = all_oh.dropna() # only want to save patients that we have all data
    
    # merge old ID / new ID into a single dictionary
    # fine to have same patients represented twice (once by both old and new)
    # because patient call will only happen on one or the other
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
    
    exclude += ['018_011'] # missing dose file
    
    # 018_019 fixed, just need to generate arrays
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
    
    val_data = next(val_batch) # generates entire validation dataset
    
    # is this necessary? should it be modifiable?
    def lr_schedule(epoch,lr):
        if epoch < 10:
            return lr
        elif 10 <= epoch < 20:
            return lr*0.9
        else:
            return max(lr*0.5, 0.00001)
        
    callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    
    if config.getboolean('data_settings','withmask',fallback=True):
        channels = 3
    else:
        channels = 2
    
    model = get_model(config)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.getfloat('model_settings', 'initial_learnrate')
            ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(),
                 tf.keras.metrics.FalseNegatives(),
                 tf.keras.metrics.FalsePositives()]
        )
    model.summary()
    history = model.fit(
        x=batch_input,
        steps_per_epoch=math.floor(
            dataset_size/config.getint('model_settings','batch_size')
            ),
        validation_data=val_data,
        epochs=config.getint('model_settings','epochs'),
        verbose=1,
        callbacks=[callback])
    
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
    
    with open(os.path.join(new_run_dest,"val_patients.txt"),"w+") as f:
        for pt in val_pts:
            f.write(pt + "\n")
        f.close()
        
    tf.keras.utils.plot_model(
        model,
        to_file=os.path.join(new_run_dest,"model_plot.png"),
        show_shapes=True
        )

if __name__ == "__main__":
    import argparse
    # IDE = True
    
    # if IDE is False:
    #     cli = argparse.ArgumentParser()
    #     cli.add_argument("configpath")
    #     clargs = cli.parse_args()
    #     config = configparser.ConfigParser()
    #     config.read(clargs.configpath)
    # elif IDE is True:
    #     config = IDE_config()
    # run(config)