# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 00:27:37 2023

@author: johna
"""

import Preprocessor
import PatientArray

import os
import pydicom
testdir = r"D:\H_N\017_055"
filepaths = [os.path.join(testdir,file) for file in os.listdir(testdir) if file.startswith("CT")]
files = [pydicom.dcmread(file) for file in filepaths]
dosefile = pydicom.dcmread(r"D:\H_N\017_055\RD.017_055.56-70.dcm")
ssfile = pydicom.dcmread(r"D:\H_N\017_055\RS.017_055.CT_1.dcm")

test = PatientArray.PatientCT(files)
test.rescale(2.5)
dose = PatientArray.PatientDose(dosefile)
mask_l = PatientArray.PatientMask(test,ssfile,"Parotid (Left)")
mask_r = PatientArray.PatientMask(test,ssfile,"Parotid (Right)")
mask_l.join(mask_r)
masks = mask_l
dose.align_with(test)

prepper = Preprocessor.Preprocessor()
prepper.attach((test,dose,masks))