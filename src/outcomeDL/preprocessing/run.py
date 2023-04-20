# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 00:27:37 2023

@author: johna
"""

import argparse

import Preprocessor as prep
import PatientArray as arrayclass
from _preprocess_util import find_parotid_info

import os
import pydicom

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessor: DICOM to HDF5')
    
    # Add the arguments
    parser.add_argument(
        'input_directory', type=str, help='Directory to load files from'
        )
    parser.add_argument(
        'destination', type=str, help="Filepath to save output to")
    parser.add_argument(
        '-px',
        '--pixel_size', 
        type=float, 
        default=2, 
        help='Desired pixel size of the eventual arrays')
    parser.add_argument('-pl', '--parotid_l', action='store_true')
    parser.add_argument('-pr', '--parotid_r', action='store_true')
    parser.add_argument(
        '-bb',
        '--bounding_box_size',
        nargs='?',
        const=None,
        default=None,
        type=str, 
        help='Bounding box size in 3 dimensions (e.g. "10,20,30")'
        )
    parser.add_argument(
        '-c', 
        '--center_of_mass', 
        action='store_true', 
        default=False,
        help='Center bounding box on organ mask center-of-mass'
        )
    
    # Parse the arguments
    args = parser.parse_args()
    if args.bounding_box_size is not None:
        boxed = True
        boxsize = args.bounding_box_size.split(",")
        boxsize = tuple(map(int, boxsize))
    else:
        boxed = False
        boxsize = None
    root = args.input_directory
        
    # Get the files - this assumes no "junk files", clean directories
    dcms = [
        pydicom.dcmread(os.path.join(root, file))
        for file in os.listdir(root)
        if file.endswith(".dcm")
        ]
    ct_files = []
    dose_files = []
    ss_files = []
    for dcm in dcms:
        if dcm.Modality == "CT":
            ct_files.append(dcm)
        elif dcm.Modality == "RTDOSE":
            dose_files.append(dcm)
        elif dcm.Modality == "RTSTRUCT":
            ss_files.append(dcm)
    
    if len(dose_files) == 1:
        dose_files = dose_files[0]
    if len(ss_files) > 1:
        raise Exception("Only one structure set file permitted in source dir.")
    ss = ss_files[0]
    
    ct_arr = arrayclass.PatientCT(ct_files)
    ct_arr.rescale(args.pixel_size)
    
    dose_arr = arrayclass.PatientDose(dose_files)
    dose_arr.align_with(ct_arr)
    
    # I don't love how the ROIs are passed yet, it's too parotid-specific
    # I will need to rewrite it before it can be used for pharyngeal
    mask_arr = None
    if args.parotid_r:
        roi_name, roi_num = find_parotid_info(ss,"r")
        temp = arrayclass.PatientMask(ct_arr, ss, roi_name)
        mask_arr = temp
    if args.parotid_l:
        roi_name, roi_num = find_parotid_info(ss, "l")
        temp = arrayclass.PatientMask(ct_arr, ss, roi_name)
        if mask_arr is None:
            mask_arr = temp
        else:
            mask_arr.join(temp)
            
    prepper = prep.Preprocessor()
    prepper.attach([ct_arr, dose_arr, mask_arr])
    
    # TODO - Prepare code for augmentation, figure out how to interface 
    # augmentation with CLI and argparse
    
    prepper.save(
        args.destination,
        boxed=boxed,
        boxshape=boxsize,
        maskcentered=args.center_of_mass
        )