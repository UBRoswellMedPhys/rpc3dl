# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 19:07:08 2023

@author: johna
"""

import os
import h5py
import numpy as np
import pandas as pd
import random
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from _utils import window_level, get_unique_values, rebuild_mask

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
            
            
class InputGenerator_v2:
    def __init__(self,root,time,windowlevel=(400,50),normalize=True,
                 ipsicontra=True):
        # right now the thing hardcodes time cutoff at 90 days
        self.root_dir = root
        self.files = [file for file in os.listdir(root) if file.endswith('.h5')]
        self.pt_char_db = pd.DataFrame()
        self.time = time
        self.days_cutoff = 90 # determines early vs late
        self.call_mode = 'train'
        self.mask_settings = {'parotid_r':True,'parotid_l':True,'ptv':False,'merge':True}
        self.call_index = -1
        self.batch_size = None # used to ensure even batches
        self.windowlevel = windowlevel
        self.normalize = normalize
        self.ipsicontra = ipsicontra
        self.set_endpoint() # sets default endpoint settings for xero
        self.scout_files()
        
    def set_endpoint(self,category='dry_mouth',threshold=3,mode='any'):
        # only supported mode right now is any cause i don't have time or
        # need to implement others
        self.lbl_cat = category
        self.lbl_threshold = threshold
        self.lbl_mode = mode
        
    def evaluate_surveys(self,df):
        result = {'early':None,'late':None,'baseline':None}
        early = df[
            (df['time_since_RT_end']>=-5)&(df['time_since_RT_end']<=self.days_cutoff)
            ]
        late = df[df['time_since_RT_end']>self.days_cutoff]
        baseline = df[(df['RT_duration'] + df['time_since_RT_end']).abs() < 10]
        subsets = {'early':early,'late':late,'baseline':baseline}
        for time in ['early','late','baseline']:
            if len(subsets[time]) == 0 or subsets[time][self.lbl_cat].isna().all():
                # if no entries exist for time window or only blank entries
                result[time] = 'invalid'
            elif (subsets[time][self.lbl_cat] >= self.lbl_threshold).any():
                result[time] = 'pos'
            else:
                # if we know it's valid and we know it's not pos, must be neg
                result[time] = 'neg'
        return result
            
        
    def scout_files(self):
        # sorts files into their respective classes. useful for getting a val
        # split that is somewhat class balanced
        # self.pos = []
        # self.neg = []
        # self.invalid = []
        self.early = {'pos':[],'neg':[],'invalid':[]}
        self.late = {'pos':[],'neg':[],'invalid':[]}
        self.baseline = {'pos':[],'neg':[],'invalid':[]}
        self.pt_char_settings = {}
        for file in self.files:
            with h5py.File(os.path.join(self.root_dir,file),'r') as f:
                if 'surveys' not in f.keys():
                    print("Bypassing {}, no surveys".format(file))
                    self.files.remove(file)
                    continue
                surveys = f['surveys'][...].astype(str)
                surv_fields = f['surveys'].attrs['fields'].astype(str)
                df = pd.DataFrame(columns=surv_fields,data=surveys)
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col],errors='coerce')
                eval_result = self.evaluate_surveys(df)
                for time in ['early','late','baseline']:
                    getattr(self,time)[eval_result[time]].append(file)
                
                # TODO - double check below
                chars = pd.DataFrame(
                    columns=f['pt_chars'].attrs['fields'].astype(str),
                    index=[file],
                    data=f['pt_chars'][...].astype(str).reshape(1,-1)
                    )
                self.pt_char_db = pd.concat(
                    [self.pt_char_db,chars]
                    )
        for field in self.pt_char_db.columns:
            if (self.pt_char_db[field]=='nan').all():
                self.pt_char_db.drop(columns=[field],inplace=True)
                continue
            if field not in self.pt_char_settings:
                self.pt_char_settings[field] = False
    
    def build_encoders(self):
        uniques = get_unique_values(self.pt_char_db,delimiter="|")
        self.encoders = {}
        for field in uniques.keys():
            fit_to = uniques[field]
            if 'nan' in fit_to:
                fit_to.remove('nan')
            if len(fit_to) == 1:
                continue
            try:
                # check to see if data can fit to numeric, if so no OHE needed
                fit_to = np.array(fit_to,dtype=np.float32)
                fit_to = fit_to[fit_to!=np.inf]
                fit_to = fit_to.reshape(-1,1)
            except ValueError:
                fit_to = np.array(fit_to).reshape(-1,1)
            if fit_to.dtype == np.float32:
                self.encoders[field] = MinMaxScaler()
            else:
                self.encoders[field] = OneHotEncoder(
                    sparse_output=False,handle_unknown='ignore'
                    )
            self.encoders[field].fit(fit_to)
    
    def pt_char_len(self):
        length = 0
        for field in self.pt_char_settings.keys():
            if self.pt_char_settings[field] is True:
                if isinstance(self.encoders[field],OneHotEncoder):
                    length += len(self.encoders[field].categories_[0])
                elif isinstance(self.encoders[field],MinMaxScaler):
                    length += 1
        return length
    
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
        pos = getattr(self,self.time)['pos']
        neg = getattr(self,self.time)['neg']
        random.seed(seed)
        random.shuffle(pos)
        random.shuffle(neg)
        i = 0
        while True:
            if len(self.test) < num_test:
                self.test.append(pos[i])
                self.test.append(neg[i])
                i += 1
                continue
            if len(self.val) < num_val:
                self.val.append(pos[i])
                self.val.append(neg[i])
                i += 1
                continue
            self.train += pos[i:]
            self.train += neg[i:]
            break
        random.shuffle(self.train)
        
    def load_patient(self,file,consider_augments=False):
        with h5py.File(os.path.join(self.root_dir,file),'r') as f:
            # first we assemble the volumetric data
            ct = f['ct'][...]
            dose = f['dose'][...]
            masks = []
            for roi in self.mask_settings.keys():
                if roi == 'merge':
                    continue
                if self.mask_settings[roi] is True:
                    masks.append(rebuild_mask(f,roi))
            if self.mask_settings['merge']:
                mask = np.zeros_like(ct)
                for m in masks:
                    mask += m
                Xvol = np.stack([ct,dose,mask],axis=-1)
            else:
                Xvol = np.stack([ct,dose] + masks,axis=-1)
                
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
                    
            # next we handle the non-volume data
            # currently do not support multiple entrypoints, it's all returned as one mass
            pt_char_fields = f['pt_chars'].attrs['fields'].astype(str)
            transformed = []
            for field,setting in self.pt_char_settings.items():
                if setting is False:
                    continue
                value = f['pt_chars'][np.where(pt_char_fields==field)].astype(str)
                values = value[0].split("|") # hardcoded delimiter
                conversion = True if isinstance(self.encoders[field],MinMaxScaler) else False
                if conversion:
                    transformed += [
                        self.encoders[field].transform(
                            np.array(val,dtype=np.float32).reshape(-1,1)
                            ) for val in values
                        ]
                else:
                    transformed += [
                        self.encoders[field].transform(
                            np.array(val).reshape(-1,1)
                            ) for val in values
                        ]
            
            if len(transformed) == 0:
                Xnonvol = None
            elif len(transformed) == 1:
                Xnonvol = transformed[0]
            else:
                Xnonvol = np.concatenate(transformed,axis=1)
                
            # finally, retrieve the label. this was pre-scouted.
            reference = getattr(self,self.time)
            if file in reference['pos']:
                Y = 1
            elif file in reference['neg']:
                Y = 0
            else:
                raise ValueError("Patient does not have valid label")
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
        support_length = self.pt_char_len()
        if support_length > 0:
            supportsig = tf.TensorSpec((support_length,),dtype=tf.float32)
        else:
            supportsig = None
        
        x_sigs = (basesig, supportsig) if supportsig is not None else basesig
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
            if xnonvol is not None:
                X_ret = (xvol,xnonvol)
            else:
                X_ret = xvol
            yield X_ret, y
            
            
if __name__ == "__main__":
    test = InputGenerator_v2(r"D:\newdata","early")
    test.build_encoders()