# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 02:35:29 2023

@author: johna
"""

import os
import pydicom
import numpy as np

import _preprocess_util as util
from PatientArray import PatientCT, PatientDose, PatientMask

BOX = (50, 128, 128)
SINGLE = False

SOURCE_DIR = r"D:\H_N\017_055"
DEST_DIR = r"D:\H_N\017_055"

# assumption here is that files have already been cleaned

filepaths = [os.path.join(SOURCE_DIR,file) for file in os.listdir(SOURCE_DIR)]
files = [pydicom.dcmread(filepath) for filepath in filepaths]
ct_files = []
dose_files = []
ss_files = []
for file in files:
    if file.Modality == "CT":
        ct_files.append(file)
    elif file.Modality == "RTDOSE":
        dose_files.append(file)
    elif file.Modality == "RTSTRUCT":
        ss_files.append(file)

if len(dose_files) == 1:
    dose_files = dose_files[0]
if len(ss_files) == 1:
    ss_files = ss_files[0]
else:
    raise Exception("Can only have one structure set file")

ct = PatientCT(ct_files)
ct.rescale(2) # use pixel size 2,2

dose = PatientDose(dose_files)
dose.align_with(ct) # fit dose array to ct

par_l_name, par_l_num = util.find_parotid_info(ss_files,"l")
par_r_name, par_r_num = util.find_parotid_info(ss_files,"r")
mask_l = PatientMask(ct,ss_files,par_l_name)
mask_r = PatientMask(ct,ss_files,par_r_name)

if SINGLE:
    mask_l.join(mask_r)
    masks = mask_l
    final = np.stack((
        ct.bounding_box(BOX),
        dose.bounding_box(BOX),
        masks.bounding_box(BOX)
        ), axis=-1)
    with open(os.path.join(DEST_DIR,"array.npy"),"wb") as f:
        np.save(f, final)
        f.close()
elif not SINGLE:
    left_final = np.stack((
        ct.bounding_box(BOX, center=mask_l.com),
        dose.bounding_box(BOX, center=mask_l.com),
        mask_l.bounding_box(BOX, center=mask_l.com)
        ), axis=-1)
    right_final = np.stack((
        ct.bounding_box(BOX, center=mask_r.com),
        dose.bounding_box(BOX, center=mask_r.com),
        mask_r.bounding_box(BOX, center=mask_r.com)
        ), axis=-1)
    with open(os.path.join(DEST_DIR,"left_array.npy"),"wb") as f:
        np.save(f, left_final)
        f.close()
    with open(os.path.join(DEST_DIR,"right_array.npy"),"wb") as f:
        np.save(f, right_final)
        f.close()