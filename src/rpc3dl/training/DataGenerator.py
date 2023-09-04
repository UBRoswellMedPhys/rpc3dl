# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 19:07:08 2023

@author: johna
"""

import os
import h5py
import json
import numpy as np
import pandas as pd
import random
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from _utils import window_level, get_unique_values, rebuild_mask, get_survival
from data_augmentation import zoom, rotate, shift
            
class InputGenerator_v2:
    
    # class method allows instantiation of generator from a config file
    @classmethod
    def from_config(cls, config):
        if isinstance(config,str):
            with open(config,"r") as f:
                config = json.load(f)
        gen = cls(
            config['root'],
            config['time'],
            windowlevel=config.get('windowlevel',(400,50)),
            normalize=config.get('normalize',True),
            ipsicontra=config.get('ipsicontra',True),
            call_augments=config.get('augment',True)
            )
        gen.pt_char_settings.update(config['pt_char_settings'])
        gen.build_encoders()
        gen.batch_size = config.get('batch_size',20)
        gen.build_splits(
            config['splitseed'],
            val=config['splitsize_valtest'][0],
            test=config['splitsize_valtest'][1],
            kfolds=config.get('kfolds',None)
            )
        if config.get('preload',False) is True:
            gen.preload()
        return gen
    
    def __init__(self,root,time,windowlevel=(400,50),normalize=True,
                 ipsicontra=True,call_augments=True,endpoint='xerostomia'):
        # right now the thing hardcodes time cutoff at 90 days
        self.root_dir = root
        self.files = [file for file in os.listdir(root) if file.endswith('.h5')]
        self.pt_char_db = pd.DataFrame()
        self.time = time
        self.days_cutoff = 90 # determines early vs late
        self.call_mode = 'train'
        self.call_augments = call_augments
        self.endpoint = endpoint
        if endpoint == 'xerostomia':
            self.mask_settings = {
                'parotid_r':True, 'parotid_l':True, 'ptv':False, 'merge':True
                }
        elif endpoint == 'survival':
            self.mask_settings = {
                'parotid_r':False, 'parotid_l':False, 'ptv':True, 'merge':False
                }
            self.time = 'survival' # this is silly but because of the way it's structured it's an easier shortcut
        self.call_index = -1
        self.batch_size = None # used to ensure even batches
        self.windowlevel = windowlevel
        self.normalize = normalize
        self.ipsicontra = ipsicontra
        self.set_endpoint() # sets default endpoint settings for xero
        self.scout_files()
        
        
    def set_endpoint(self,category='dry_mouth',threshold=3,mode='any'):
        """
        This method is called during init. This stores a few variables that guide
        the survey handler for when the endpoint is determined by QOL surveys.
        The default values for the arguments here are what I am using for
        xerostomia work and there is no need to provide other values, so the
        init function calls it without providing alternate values.
        
        If the endpoint is survival, this function does not do anything. It's 
        still called during init but it has no effect on the process.
        """
        self.lbl_cat = category
        self.lbl_threshold = threshold
        self.lbl_mode = mode
        
    def evaluate_surveys(self,df):
        """
        Called during scout files. Processes a single patient's QOL follow up
        surveys DataFrame and returns the a dictionary of results by time
        window (baseline, early, late). Uses the fields "RT_duration" and
        "time_since_RT_end" to determine timing. The "time_since_RT_end" field
        is not native to RedCap data, rather, it is generated independently
        prior to initial data processing, using the survey date against the
        RT end date (dates are potentially PII so they must be erased during
        data processing, this allows the necessary information to persist).
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of a single patient's survey responses.
            
        Returns
        -------
        result : dict
            Dictionary that maps time windows (baseline, early, late) to label
            values (pos, neg, invalid).
        """
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
        """
        Initial examination of data. Evaluates the endpoint to categorize each
        patient as positive, negative, or invalid (unsufficient information
        to categorize).
        
        It also collects the nonvolume patient characteristic data and loads
        it all into an attribute pt_char_db, which is used to configure
        encoders later.
        
        Counterintuitive note: for the survival endpoint, positive label
        indicates DEATH within 2 years. Negative means survival to two years.
        """
        self.early = {'pos':[],'neg':[],'invalid':[]}
        self.late = {'pos':[],'neg':[],'invalid':[]}
        self.baseline = {'pos':[],'neg':[],'invalid':[]}
        self.survival = {'pos':[],'neg':[],'invalid':[]}
        self.pt_char_settings = {}
        for file in self.files:
            with h5py.File(os.path.join(self.root_dir,file),'r') as f:
                if 'pt_chars' not in f.keys():
                    print("Bypassing {}, no pt_chars".format(file))
                    self.files.remove(file)
                    continue
                if self.endpoint == 'xerostomia':
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
                survival_label = get_survival(f,cutoff=730) # gets survival info at cutoff time
                if survival_label == 1:
                    self.survival['pos'].append(file)
                elif survival_label == 0:
                    self.survival['neg'].append(file)
                else:
                    self.survival['invalid'].append(file)
                
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
        """
        This method fits all the necessary encoders for non-volume data. For
        categorical data, each field is converted into one-hot encoded format.
        For numerical data, a MinMaxScaler is applied.
        
        This function depends on the pt_char_db attribute, which is built
        during scout_files(), but scout_files is called during init so the
        user should not ever have to track or worry about this dependency.
        """
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
                try:
                    self.encoders[field] = OneHotEncoder(
                        sparse_output=False,handle_unknown='ignore'
                        )
                except TypeError:
                    self.encoders[field] = OneHotEncoder(
                        sparse=False,handle_unknown='ignore'
                        )
            if field == 'HPV status':
                # override configuratino for this field, we want to remove
                # "Unknown" as an option
                fit_to = [['Positive'],['Negative']]
            self.encoders[field].fit(fit_to)
    
    @property
    def pt_char_len(self):
        # length of the 1D array output of nonvolume data for each patient
        length = 0
        for field in self.pt_char_settings.keys():
            if self.pt_char_settings[field] is True:
                if isinstance(self.encoders[field],OneHotEncoder):
                    length += len(self.encoders[field].categories_[0])
                elif isinstance(self.encoders[field],MinMaxScaler):
                    length += 1
        return length
    
    def pt_char_desc(self):
        """
        Very important method for understanding the non-volume patient
        characteristic encoding. Loops through all the active fields of
        pt_char_settings and draws out description of each position. For
        MinMaxScaler encodings, it simply uses the field name, but for OHE
        encodings it expands into a list of the form ['field: option 1',
        'field: option 2',...]. Once all iteration is complete, the return
        is a list that matches up descriptions to positions in the 1D array
        that nonvolume data is served as for each patient.
        """
        desc = []
        for field in self.pt_char_settings.keys():
            if self.pt_char_settings[field] is True:
                if isinstance(self.encoders[field],OneHotEncoder):
                    to_add = list(self.encoders[field].categories_[0])
                    to_add = ['{}: {}'.format(field,x) for x in to_add]
                    desc += to_add
                elif isinstance(self.encoders[field],MinMaxScaler):
                    desc.append(field)
        return desc
    
    def __len__(self):
        # total number of valid staged files
        return len(self.files) - len(getattr(self,self.time)['invalid'])
      
    @property
    def train_ceiling(self):
        """
        The tf.Dataset from generator structure doesn't play nice with leftover
        partial batches at the end of epochs, so this value is used to get a
        clean ceiling that ensures complete batches. Obviously this leaves
        some remainder of patients out of any given epoch, however, since the
        generator shuffles the list at each reset, over multiple epochs every
        data element is expected to be included at some point.
        """
        if self.batch_size is None:
            c = len(self.train)
        else:
            c = len(self.train) - (len(self.train) % self.batch_size)
        return c
    
    def build_splits(self,seed,val,test=0.0,kfolds=None):
        """
        Method to identify which files in self.root_dir should be assigned to
        which splits. It relies on the scouted files (which happens during
        init, always) label category to ensure a balanced val/test set.
        
        Parameters
        ----------
        seed : int
            Seed to use for randomization. Essential for reproducability.
        val : float
            Decimal from 0.0-1.0 to represent how much of the total dataset to
            reserve as validation data. Class balance is enforced.
        test : float or int, optional
            Decimal from 0.0-1.0 to represent how much of the total dataset to
            reserve as test data. Class balance is enforced. Default is 0.0.
            If kfolds is not None, then this argument indicates which fold to
            use as the test fold.
        kfolds : int
            Number of folds to divide dataset into.
        """
        if kfolds is not None:
            assert isinstance(test,int), "When using kfolds, test must be int"
        self.kfolds = kfolds # save info for config saving later
        self.splitseed = seed
        self.splitsize_valtest = (val, test)
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
        if kfolds is None:
            print("Building splits: {} test, {} val, {} train".format(
                num_test, num_val, len(self) - num_test - num_val
                ))
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
        elif kfolds is not None:
            print("Building splits, using {} kfolds...".format(kfolds))
            folds = [np.concatenate((x,y),axis=0) for x,y in zip(
                np.array_split(pos,kfolds), np.array_split(neg,kfolds)
                )]
            self.test = folds.pop(test)
            restfolds = np.concatenate(folds,axis=0)
            random.shuffle(restfolds)
            self.val = restfolds[:num_val]
            self.train = restfolds[num_val:]
        
    def load_patient(self,file,consider_augments=False):
        """
        Methods which loads patient data directly from a file stored on disk.
        
        Parameters
        ----------
        file : str
            Filename to load from. In normal use, this will be handled by
            wrapper functions, which will pull filenames out of prepopulated
            lists that are created during the scout_files and build_splits
            methods. However, this method can be invoked directly if desired,
            which can be useful for debugging.
        consider_augments : bool
            This determines whether augmented data is loaded. The augmented
            versions of the the volumetric data must be prepared in advance
            and saved into the same HDF5 file. Hardcoded to only consider three
            augmented versions and selects the original version 1/3 of the time
            and a randomly selected augmented version 2/3 of the time.
            
        Returns
        -------
        Xvol : np.ndarray
            4-dimensional array representing the volumetric input data
        Xnonvol : np.ndarray or None
            If patient characteristics are being included, a one-hot-encoded
            array representation of these traits is returned here. Otherwise,
            this returns None.
        Y : int
            0 or 1, this is the target data.
        """
        with h5py.File(os.path.join(self.root_dir,file),'r') as f:
            # first we assemble the volumetric data
            if consider_augments is False:
                volsource = f
            elif consider_augments is True:
                # generate random float between 0.0 and 1.0
                proc = np.random.random()
                if proc > 0.333333:
                    aug_idx = np.random.choice([0,1,2])
                    volsource = f['augment{}'.format(aug_idx)]
                else:
                    volsource = f
            
            ct = volsource['ct'][...]
            dose = volsource['dose'][...]
            masks = []
            for roi in self.mask_settings.keys():
                if roi == 'merge':
                    continue
                if self.mask_settings[roi] is True:
                    masks.append(rebuild_mask(volsource,roi))
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
                    temp = np.zeros(shape=(1,len(self.encoders[field].categories_[0])))
                    for val in values:
                        temp += self.encoders[field].transform(
                            np.array(val).reshape(-1,1)
                            )
                    transformed += [temp]
            
            if len(transformed) == 0:
                Xnonvol = None
            elif len(transformed) == 1:
                Xnonvol = transformed[0]
            else:
                Xnonvol = np.concatenate(transformed,axis=1)
                
            if Xnonvol is not None:
                Xnonvol = np.squeeze(Xnonvol)
                Xnonvol[np.isnan(Xnonvol)] = 0
                
            # finally, retrieve the label. this was pre-scouted.
            reference = getattr(self,self.time)
            if file in reference['pos']:
                Y = 1
            elif file in reference['neg']:
                Y = 0
            else:
                raise ValueError("Patient does not have valid label")
        return Xvol, Xnonvol, Y
    
    def load_all(self,which,aug=0):
        """
        Method to load an entire data group. This requires that splits have
        already been built.
        
        This function now supports including some augmentation. This is mostly
        useful for if you have a smaller dataset, don't want to configure your
        model to train from a generator (so can't do on-the-fly augmentation),
        but still want augmented data.

        Parameters
        ----------
        which : str
            String designator of which group to load in its entirety. Supported
            values are "train", "val", and "test".
        aug : int, optional
            This is the number of copies of each patient you'd like to produce
            with augmentation. The default is 0, which means you will only
            receive the original unmodified data.

        Returns
        -------
        Tuple of X_ret, Y
            This should be formatted acceptably to feed directly into your
            model via fit() or predict() or as the validation_data argument.

        """
        filelist = getattr(self,which)
        Xvol = []
        Xnonvol = []
        Y = []
        for file in filelist:
            xvol, xnonvol, y = self.load_patient(file,consider_augments=False)
            Xvol.append(xvol)
            if self.pt_char_len != 0:
                Xnonvol.append(xnonvol)
            Y.append(y)
            if aug > 0:
                for i in range(aug):
                    temp = np.copy(xvol)
                    functions = [zoom, rotate, shift]
                    choices = np.random.choice([0,1,2],size=2,replace=False)
                    for i in choices:
                        temp = functions[i](temp)
                    Xvol.append(temp)
                    if self.pt_char_len != 0:
                        Xnonvol.append(xnonvol)
                    Y.append(y)
                    
        if len(Xnonvol) > 0:
            X_ret = (np.array(Xvol),np.array(Xnonvol))
        else:
            X_ret = np.array(Xvol)
        return X_ret, np.array(Y)
    
    def preload(self):
        """
        Call to preload training data into active memory. This modifies the
        behavior of all future training data calls, in that they will no longer
        stream the data from file but instead will retrieve from the preloaded
        data.
        """
        print("Preloading training data...")
        self.loaded_train = self.load_all('train')
        print("Done. Calling the generator will now fetch from preload.")
    
    @property
    def output_sig(self):
        """
        When instantiating a tf.Dataset from a generator, it needs to know the
        output signature. This is a quick utility function that constructs
        that signature in appropriate TensorSpec format based on how the
        instance is configured.
        """
        if self.endpoint == 'xerostomia':
            basesig = tf.TensorSpec((40,128,128,3),dtype=tf.float32)
        elif self.endpoint == 'survival':
            basesig = tf.TensorSpec((60,128,128,3),dtype=tf.float32)
        support_length = self.pt_char_len
        if support_length > 0:
            supportsig = tf.TensorSpec((support_length,),dtype=tf.float32)
        else:
            supportsig = None
        
        x_sigs = (basesig, supportsig) if supportsig is not None else basesig
        y_sig = tf.TensorSpec(shape=(),dtype=tf.int32)
        return (x_sigs, y_sig)

    
    def fetch_patient(self,idx,consider_augments=False):
        """
        Method that is the mirror of load_patient, except for it fetches a
        patient out of the preloaded array. If augments are turned on, this
        function performs them on-the-fly.

        Parameters
        ----------
        idx : int
            Number index of the patient to retrieve.
        consider_augments : bool
            Whether to perform on-the-fly augmentation. The default is False.

        Returns
        -------
        X : np.ndarray
            Volume data array for the patient.
        Xnonvol : np.ndarray or None
            Array of one-hot-encoded patient characteristics if active, if
            none are active then returns None for this value.
        Y : int
            Target label of patient.
        """
        if self.pt_char_len > 0:
            X = self.loaded_train[0][0][idx,...]
            Xnonvol = self.loaded_train[0][1][idx,:]
        else:
            X = self.loaded_train[0][idx,...]
            Xnonvol = None
        Y = self.loaded_train[1][idx]
        if consider_augments is True:
            if np.random.random() < 0.75:
                functions = [zoom, rotate, shift]
                choices = np.random.choice([0,1,2],size=2,replace=False)
                for i in choices:
                    X = functions[i](X)
        return X, Xnonvol, Y
    
            
    def __call__(self):
        """
        In order to use an object to instantiate a tf.Dataset from a generator,
        the object's __call__ must actually return a generator. So this
        structure iterates through the training patient set one at a time
        and returns data. This does NOT handle batching - the tf.Dataset
        class has a batch method that we use in the training script. It DOES
        handle epochs - there is a counter attribute (self.call_index) that
        monitors the iteration through the training set. When you reach the
        end of the training set, the next call will detect that, reset the
        counter, shuffle the training dataset list, and start over. This
        means that the generator returned by __call__ is an infinite generator,
        it never runs out of data, because it continuously loops.
        
        You can manually invoke the call to step through data loading one at a
        time (for instance, to verify that the augments are working as intended)
        using "next(gen())" where gen is your configured InputGenerator_v2 
        class object. If you need to repeat loading a same-patient, you'll need
        to manually tick down the counter again after each call. Keep in mind
        that end-of-epoch shuffling is irreversible as it is not seeded.
        """
        while True:
            self.call_index += 1
            if self.call_index >= self.train_ceiling:
                # if index exceeds ceiling, reset index and reshuffle data
                self.call_index = 0
                random.shuffle(self.train)
                if hasattr(self,'loaded_train'):
                    indices = np.arange(self.loaded_train[1].shape[0])
                    np.random.shuffle(indices)
                    if isinstance(self.loaded_train[0],tuple):
                        self.loaded_train = (
                            (self.loaded_train[0][0][indices,...],self.loaded_train[0][1][indices,:]),
                            self.loaded_train[1][indices]
                            )
                    else:
                        self.loaded_train = (
                            self.loaded_train[0][indices,...],
                            self.loaded_train[1][indices]
                            )
            
            # if data's preloaded, pull from that, otherwise, load from file
            if hasattr(self,'loaded_train'):
                xvol, xnonvol, y = self.fetch_patient(
                    self.call_index,consider_augments=self.call_augments
                    )
            else:
                xvol, xnonvol, y = self.load_patient(
                    self.train[self.call_index], consider_augments=self.call_augments
                    )
                
            if xnonvol is not None:
                X_ret = (xvol,xnonvol)
            else:
                X_ret = xvol
            yield X_ret, y
            
    def export_config(self,fname):
        """
        Utility function to save the configuration to a JSON file. This file
        can then instantiate an identical version of the generator using the
        from_generator classmethod.

        Parameters
        ----------
        fname : str
            Filename to store config to.

        """
        config = {}
        config['root'] = self.root_dir
        config['time'] = self.time
        config['windowlevel'] = self.windowlevel
        config['normalize'] = self.normalize
        config['ipsicontra'] = self.ipsicontra
        config['augment'] = self.call_augments
        config['pt_char_settings'] = self.pt_char_settings
        config['batch_size'] = self.batch_size
        config['splitseed'] = self.splitseed
        config['kfolds'] = self.kfolds
        config['splitsize_valtest'] = self.splitsize_valtest
        if hasattr(self,'loaded_train'):
            config['preload'] = True
        with open(fname,"w+") as f:
            json.dump(config,f,indent=4)
            
if __name__ == "__main__":
    test = InputGenerator_v2(r"E:\newdata","early")
    import pt_char_lookups
    active = pt_char_lookups.groups['Conditions']
    PT_CHAR_SETTINGS = {field : True for field in active}
    test.pt_char_settings.update(PT_CHAR_SETTINGS)
    test.build_encoders()
    test.build_splits(98,val=0.1,test=0.1)