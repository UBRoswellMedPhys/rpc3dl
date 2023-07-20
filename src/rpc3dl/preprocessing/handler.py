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
        for col in surveys.columns:
            if any(('date' in col.lower(),'timestamp' in col.lower())):
                surveys = surveys.drop(columns=[col])        
        
        subset = surveys[surveys['mrn']==self.patient_id]
        if len(subset) == 0:
            # kill process if no surveys match
            return None
        subset = subset.drop(columns=['mrn'])
        self.surveys = subset.to_numpy()
        self.survey_fields = subset.columns
        
    def get_pt_chars(self,pc_file):
        if self.patient_id is None:
            raise Exception("Cannot fetch pt_chars without patient ID")
        pc_file.index = pc_file.index.astype(str)
        if str(self.patient_id) in pc_file.index:
            self.pt_chars = pc_file.loc[str(self.patient_id)].to_numpy()
            self.pt_char_fields = pc_file.columns
            
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
            mask_arrays = [mask.array for mask in self.mask]
        elif boxed is True:
            if maskcentered is True:
                if len(self.mask) == 1:
                    center = self.mask[0].com
                elif len(self.mask) > 1:
                    combined = self.mask[0].array
                    for i in range(1,len(self.mask)):
                        combined += self.mask[i].array
                    combined[combined > 1] = 1
                    livecoords = np.argwhere(combined)
                    center = np.sum(livecoords,axis=0) / len(livecoords)
            else:
                center = None
            data = [
                self.ct.bounding_box(shape=boxshape,center=center),
                self.dose.bounding_box(shape=boxshape,center=center)
                ]
            mask_arrays = [
                mask.bounding_box(shape=boxshape,center=center) for mask in self.mask
                ]
        mask_names = [mask.roi_name for mask in self.mask]
        
        with h5py.File(fname,"a") as file:
            file.create_dataset('ct',data=data[0])
            file.create_dataset('dose',data=data[1])
            for mask,name in zip(mask_arrays,mask_names):
                file.create_group(name)
            
            
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
                    file['pt_chars']
            if hasattr(self,'surveys'):
                if 'surveys' not in file.keys():
                    file.create_group("surveys",data=self.surveys)
                    file['surveys'].attrs['fields'] = self.survey_fields
                    
            
                
                
def get_dataset_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys