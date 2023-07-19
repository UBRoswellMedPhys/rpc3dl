# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 19:07:08 2023

@author: johna
"""

import os
import h5py
import numpy as np
import random
import tensorflow as tf

from _utils import window_level

class InputGenerator:
    def __init__(self,root,time,windowlevel=(400,50),normalize=True,
                 ipsicontra=True, call_augments=True):
        self.root_dir = root
        self.files = [file for file in os.listdir(root) if file.endswith('.h5')]
        self.time = time
        self.call_mode = 'train'
        self.call_index = -1
        self.batch_size = None # used to ensure even batches
        self.windowlevel = windowlevel
        self.normalize = normalize
        self.ipsicontra = ipsicontra
        self.scout_files()
        self.call_augments = call_augments
        
    def scout_files(self):
        # sorts files into their respective classes. useful for getting a val
        # split that is somewhat class balanced
        self.pos = []
        self.neg = []
        self.invalid = []
        self.pt_char_settings = {}
        for file in self.files:
            with h5py.File(os.path.join(self.root_dir,file),'r') as f:
                if f['labels'][self.time][...] == 0:
                    self.neg.append(file)
                elif f['labels'][self.time][...] == 1:
                    self.pos.append(file)
                else:
                    self.invalid.append(file)
                for field in f['pc'].keys():
                    if field not in self.pt_char_settings:
                        self.pt_char_settings[field] = False
    
    def __len__(self):
        return len(self.files) - len(self.invalid)
    
    @property
    def train_ceiling(self):
        if self.batch_size is None:
            c = len(self.train)
        else:
            c = len(self.train) - (len(self.train) % self.batch_size)
        return c
    
    def build_splits(self,seed,val,test=0.0):
        # set up splits, val and test should be floats between 0.0 and 1.0
        num_val = round(len(self) * val)
        num_test = round(len(self) * test)
        self.train = []
        self.val = []
        self.test = []
        random.seed(seed)
        random.shuffle(self.pos)
        random.shuffle(self.neg)
        i = 0
        while True:
            if len(self.test) < num_test:
                self.test.append(self.pos[i])
                self.test.append(self.neg[i])
                i += 1
                continue
            if len(self.val) < num_val:
                self.val.append(self.pos[i])
                self.val.append(self.neg[i])
                i += 1
                continue
            self.train += self.pos[i:]
            self.train += self.neg[i:]
            break
        random.shuffle(self.train)
        
    def load_patient(self,file,consider_augments=False):
        Xnonvol = {}
        with h5py.File(os.path.join(self.root_dir,file),'r') as f:
            # clause for augment selection
            if consider_augments is True:
                # TODO - In future we may want to include ALL rather than randomly pull
                if np.random.random() < 0.65:
                    select = np.random.choice(['augment0','augment1','augment2']) #TODO - hardcoded this, naughty
                    Xvol = f[select][...]
                else:
                    Xvol = f['base'][...]
            else:
                Xvol = f['base'][...]
            
            if self.ipsicontra:
                midpoint = int(Xvol.shape[2]/2)
                if np.sum(Xvol[:,:,midpoint:,1]) > np.sum(Xvol[:,:,:midpoint,1]):
                    Xvol = np.flip(Xvol,axis=2)
            if self.windowlevel is not None:
                Xvol[...,0] = window_level(
                    Xvol[...,0],
                    window=self.windowlevel[0],
                    level=self.windowlevel[1]
                    )
            if self.normalize:
                Xvol[...,0] = (Xvol[...,0] - np.amin(Xvol[...,0])) \
                    / (np.amax(Xvol[...,0] - np.amin(Xvol[...,0])))
                Xvol[...,1] = Xvol[...,1] / 70 # voxels are repped in Gy
                    
            for field,setting in self.pt_char_settings.items():
                if setting is False:
                    continue
                Xnonvol[field] = f['pc'][field][...]
                
            Y = f['labels'][self.time][...]
        return Xvol, Xnonvol, Y
    
    def load_all(self,which):
        filelist = getattr(self,which)
        Xvol = []
        Xnonvol = {k:[] for k,v in self.pt_char_settings.items() if v is True}
        Y = []
        for file in filelist:
            xvol, xnonvol, y = self.load_patient(file)
            Xvol.append(xvol)
            for field in Xnonvol.keys():
                Xnonvol[field].append(xnonvol[field])
            Y.append(y)
        if len(Xnonvol) > 0:
            X_ret = [np.array(Xvol)]
            for charlist in Xnonvol.values():
                X_ret.append(np.array(charlist))
            X_ret = tuple(X_ret)
        else:
            X_ret = np.array(Xvol)
        return X_ret, np.array(Y)
    
    def output_sig(self):
        basesig = tf.TensorSpec((40,128,128,3),dtype=tf.float32)
        reference_sigs = {
            'age' : tf.TensorSpec((),dtype=tf.int32),
            'disease_site' : tf.TensorSpec((6,),dtype=tf.int32),
            'gender' : tf.TensorSpec((2,),dtype=tf.int32),
            'hpv' : tf.TensorSpec((4,),dtype=tf.int32),
            'm_stage' : tf.TensorSpec((4,),dtype=tf.int32),
            'n_stage' : tf.TensorSpec((8,),dtype=tf.int32),
            'race' : tf.TensorSpec((5,),dtype=tf.int32),
            'smoking' : tf.TensorSpec((3,),dtype=tf.int32),
            't_stage' : tf.TensorSpec((10,),dtype=tf.int32),
            'treatment_type' : tf.TensorSpec((6,),dtype=tf.int32)
            }
        additionalsigs = []
        for k,v in self.pt_char_settings.items():
            if v is True:
                additionalsigs.append(reference_sigs[k])
        if len(additionalsigs) > 0:
            x_sigs = [basesig] + additionalsigs
            x_sigs = tuple(x_sigs)
        else:
            x_sigs = basesig
        y_sig = tf.TensorSpec(shape=(),dtype=tf.int32)
        return (x_sigs, y_sig)
    
    def __call__(self):
        while True:
            self.call_index += 1
            if self.call_index >= self.train_ceiling:
                self.call_index = 0
                random.shuffle(self.train)
            
            xvol, xnonvol, y = self.load_patient(
                self.train[self.call_index],
                consider_augments=self.call_augments
                )
            if len(xnonvol) > 0:
                X_ret = [xvol]
                for v in xnonvol.values():
                    X_ret.append(v)
                X_ret = tuple(X_ret)
            else:
                X_ret = xvol
            yield X_ret, y
            
            
                