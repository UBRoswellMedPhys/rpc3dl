# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 15:23:55 2023

@author: johna

File to prepare a generator that loads model input data *one at a time*

Performs NO TRANSFORMS on the data. Expectation is that core changes to
data will be done on array preparation and storage, not on load. This is to
keep training time to a minimum.
"""

import os
import h5py
import pandas as pd
import numpy as np
import tensorflow as tf

"""
Assumes directory structure as follows:
    
    SOURCE_DIR
    |
    |_PT1
    |_PT2
    |_PT3
    |_PT4
    ...
    |_labels.csv
    |_patientinfo.csv
    
Naming scheme for files:
    Each patient folder will have at least one HDF5 file in its folder. This 
    file will have a 4D array that represents the patient volume and the
    CT/DOSE/MASK channels. 
    
    Base naming will be: data.h5
    
    If using the dual-channel approach, two "base" files
    will be there. In this case, the files will be named:
        data_l.h5, data_r.h5
    Note that "l" and "r" conventions are used for naming but this does not
    necessarily map to left and right - for example, it may instead be
    ipsilateral and contralateral, depending on data settings during array
    creation.
    
    Finally, if data augmentation is enabled, then we will save MULTIPLE arrays.
    This is not disk-space-efficient, but it's training time efficient, because
    performing on-the-fly augmentation on 4D arrays is extremely computationally
    taxing. In this case, there will be multiple files, named as follows:
        data_l.h5 (base file)
        data_r.h5 (base file)
        data_l_1.h5
        data_r_1.h5
        ...
        data_l_n.h5
        data_r_n.h5
    
    For this case, the generator will select a permutation of the base data
    and retrieve the file(s) that corresponds to that permutation.
"""

def data_generator(idlist,
                   root_dir,
                   ptchar=True,
                   augment=False,
                   idcol="MRN"):
    labels = pd.read_csv(os.path.join(root_dir,'labels.csv'))
    labels = labels.set_index(idcol,drop=True)
    db = pd.read_csv(os.path.join(root_dir,'patientinfo.csv'))
    db = db.set_index(idcol,drop=True)
    
    if ptchar is True:
        # if ptchar is on, we prep the lookup dict to OHE values
        # do this prior to pt loop so we only build each 
        lookups = {}
        for col in db.columns:
            try:
                # tries to cast it to int - if it works, on to next column
                db[col].astype(int)
                continue
            except ValueError:
                # if casting to int throws ValueError, proceed with str lookup
                pass
            vocab = db[col].unique().astype(str)
            lookups[col] = tf.keras.layers.StringLookup(
                vocabulary=vocab,
                output_mode='one_hot'
                )
    
    # === Prep is done, now we loop and yield ===
    for pt in idlist:
        Y = int(labels.loc[pt,'label'])
        
        datafile = os.path.join(root_dir,pt,"data.h5")
        with h5py.File(datafile,mode='r') as data:
            bilateral = data.attrs['bilateral']
            if not augment:
                select = 0
            else:
                num_augs = data.attrs['augments']
                select = np.random.choice(
                    np.arange(num_augs+1)
                    ) # add one for unaugmented copy
            if select == 0:
                load = 'main'
            else:
                load = f'aug{select}'
                
            if bilateral:
                X = [data[f'{load}_l'],data[f'{load}_r']]
            else:
                X = [data[f'{load}']]
                
        if ptchar is True:
            chars = []
            for col in db:
                if col in lookups.keys():
                    chars.append(
                        lookups[col]([str(db.loc[pt,col])])
                        )
                else:
                    chars.append(
                        tf.constant([int(db.loc[pt,col])])
                        )
            chars_tensor = tf.concat(chars,axis=0)
            X.append(chars_tensor)
            
        X = tuple(X)
        if len(X) == 1:
            X = X[0]
        yield X, Y