# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 23:04:52 2023

@author: johna
"""

import random
import numpy as np
import h5py

import rpc3dl.preprocessing.arrayclasses as arrayclass

class Preprocessor:
    
    def __init__(self,patient_id=None):
        self._ct = None
        self._dose = None
        self._mask = []
        self.patient_id = patient_id
        self.label = 99
        self.pt_chars = []
        
    def load_from_file(self,filepath):
        with h5py.File(filepath,"r") as f:
            array = f['base'][...]
            self._ct = array[...,0]
            self._dose = array[...,1]
            self._mask = array[...,2]
            self.pt_chars = f['pt_chars'][...]
            if 'patient_id' in f.attrs.keys():
                self.patient_id = f.attrs['patient_id']
        
    @property
    def ct(self):
        return self._ct
    
    @ct.setter
    def ct(self, value):
        if self._ct is None:
            self._ct = value
        else:
            raise ValueError(
                "Cannot overwrite CT array - must run erase function first"
                )
            
    @property
    def dose(self):
        return self._dose
    
    @dose.setter
    def dose(self, value):
        if self._dose is None:
            self._dose = value
        else:
            raise ValueError(
                "Cannot overwrite dose array - must run erase function first"
                )
    
    @property
    def mask(self):
       return self._mask
    
    @mask.setter
    def mask(self, value):
        if self._mask is None:
            if not isinstance(value, list):
                value = [value] # enforces that mask is always a list
            self._mask = value
        else:
            raise ValueError(
                "Cannot overwrite mask array - must run erase function first"
                )
            
    def append_mask(self,newmask):
        self._mask.append(newmask)
            
    @property
    def augmented(self):
        return any((hasattr(self.ct,'original'),
                    hasattr(self.dose,'original'),
                    hasattr(self.mask,'original')))
    
    def attach(self,patientarray):
        if not hasattr(patientarray, "__iter__"):
            patientarray = [patientarray]
        for x in patientarray:
            if isinstance(x, arrayclass.PatientCT):
                x.array = x.array.astype(np.float32)
                self.ct = x
            elif isinstance(x, arrayclass.PatientDose):
                x.array = x.array.astype(np.float32)
                self.dose = x
            elif isinstance(x, arrayclass.PatientMask):
                x.array = x.array.astype(np.int16)
                self.append_mask(x)
                
    def get_label(self,labeldf):
        if self.patient_id is None:
            raise Exception("Cannot fetch label without patient ID")
        labeldf.index = labeldf.index.astype(str)
        if str(self.patient_id) in labeldf.index:
            self.label = labeldf.loc[str(self.patient_id),'label']
            
    def populate_surveys(self,binned_survey_df):
        surveys = binned_survey_df
        surveys['mrn'] = surveys['mrn'].astype(int).astype(str)
        fields = surveys.columns[2:]
        
        
        subset = surveys[surveys['mrn']==self.patient_id]
        if len(subset) == 0:
            # kill process if no surveys match
            return None
        
        self.surveys = {}
        #f.create_group("surveys")
        self.survey_fields = np.array(fields).astype('S')
        for time in ['acute','early','late']:
            subsubset = subset[subset['bin']==time]
            if len(subsubset)==0:
                continue
            survey_array = subsubset.iloc[:,2:].to_numpy()
            self.surveys[time] = survey_array
        
    def get_pt_chars(self,pc_file):
        if self.patient_id is None:
            raise Exception("Cannot fetch pt_chars without patient ID")
        pc_file.index = pc_file.index.astype(str)
        if str(self.patient_id) in pc_file.index:
            self.pt_chars = pc_file.loc[str(self.patient_id)].to_numpy()
            
    def erase(self,mode):
        if mode.lower() == "ct":
            self._ct = None
        elif mode.lower() == "dose":
            self._dose = None
        elif mode.lower() == "mask":
            self._mask = None
        elif mode.lower() == "all":
            self._ct = None
            self._dose = None
            self._mask = None
        else:
            raise ValueError(
                "Unrecognized mode - accepts CT, Dose, Mask, or ALL"
                )
            
    def enforce_compat(self):
        # note that this presumes all three are present, true for my research
        conditions = (
            (self.ct.pixel_size == self.dose.pixel_size == self.mask.pixel_size),
            (self.ct.slice_ref == self.dose.slice_ref == self.mask.slice_ref),
            (self.ct.array.shape == self.dose.array.shape == self.mask.array.shape),
            (self.ct.patient_id == self.dose.patient_id == self.mask.patient_id)
            )
        testnames = ['Pixel Size', 'Slice Reference List','Array Shape', 'Patient ID']
        for cond, desc in zip(conditions,testnames):
            if not cond:
                raise ValueError(
                    "Array compatibility test failed - {}".format(desc)
                    )
            
    def zoom(self,maximum=0.2,exact=None):
        # establish seed
        seed = np.random.randint(0,10000)
        for subarr in ['ct','mask','dose']:
            if getattr(self,subarr) is not None:
                getattr(self,subarr).zoom(
                    max_zoom_factor=maximum,
                    seed=seed,
                    zoom_factor=exact
                    )
    
    def shift(self, maximum=0.2, exact=None):
        # establish seed
        seed = np.random.randint(0,10000)
        for subarr in ['ct','mask','dose']:
            if getattr(self,subarr) is not None:
                getattr(self,subarr).shift(
                    max_shift=maximum,
                    seed=seed,
                    pixelshift=exact
                    )
        
    def rotate(self, maximum=15, exact=None):
        # establish seed
        seed = np.random.randint(0,10000)
        for subarr in ['ct','mask','dose']:
            if getattr(self,subarr) is not None:
                getattr(self,subarr).rotate(
                    degree_range=maximum,
                    seed=seed,
                    degrees=exact
                    )
    
    def random_augment(self,num_augs=2,replace=False):
        augs = ['zoom','rotate','shift']
        aug_idx = [0,1,2]
        performed = 0
        while performed < num_augs:
            select = np.random.choice(aug_idx)
            op = augs[select]
            if replace is False:
                aug_idx.remove(select)
            getattr(self,op)() # run the augmentation
            performed += 1
                
    def reset_augments(self):
        for subarr in ['ct','mask','dose']:
            if getattr(self,subarr) is not None:
                getattr(self,subarr).reset_augments()
                
    def save(self,
             fname,
             boxed=False,
             boxshape=None,
             maskcentered=False,
             overwrite=True):
        
        # disable enforce compatibility, will revisit later
        # self.enforce_compat()
        
        # package into 4D array
        if boxed is False:
            data = [self.ct.array,self.dose.array]
            data += [mask.array for mask in self.mask]
        elif boxed is True:
            if maskcentered is True:
                center = None
                # TODO - need to fix center of mass work for multiple masks
                # center = self.mask.com
            else:
                center = None
            data = [
                self.ct.bounding_box(shape=boxshape,center=center),
                self.dose.bounding_box(shape=boxshape,center=center)
                ]
            data += [
                mask.bounding_box(shape=boxshape,center=center) for mask in self.mask
                ]
        final_array = np.stack(data,axis=-1)
        array_names = ['CT','DOSE'] + [mask.roi_name for mask in self.mask]
        
        with h5py.File(fname,"a") as file:
            if not self.augmented:
                file.create_dataset('base',data=final_array)
            else:
                ds_keys = get_dataset_keys(file)
                if 'base' in ds_keys:
                    ds_keys.remove('base')
                if 'pt_chars' in ds_keys:
                    ds_keys.remove('pt_chars')
                if 'surveys' in ds_keys:
                    ds_keys.remove('surveys')
                if len(ds_keys) == 0:
                    new_ver = 1
                else:
                    versions = [int(key.split("_")[-1]) for key in ds_keys]
                    new_ver = np.amax(versions) + 1
                file.create_dataset(
                    'augment_{}'.format(new_ver), data=final_array
                    )
            file.attrs['label'] = int(self.label)
            file.attrs['array_names'] = array_names
            if overwrite:
                if 'pt_chars' in file.keys():
                    del file['pt_chars']
                if 'surveys' in file.keys():
                    del file['surveys']
            if hasattr(self,'pt_chars'):
                if 'pt_chars' not in file.keys():
                    file.create_dataset('pt_chars', data=self.pt_chars)
            if hasattr(self,'surveys'):
                if 'surveys' not in file.keys():
                    file.create_group("surveys")
                    for time in ['acute','early','late']:
                        if time in self.surveys.keys():
                            file['surveys'].create_dataset(
                                time,data=self.surveys[time]
                                )
            
                
                
def get_dataset_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys