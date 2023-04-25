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
import pandas as pd
import pydicom

"""
Script for CLI processing of DICOM files into HDF5 files which will house the
input data for neural net training.

Here is a breakdown of the CLI arguments:
    POSITIONAL
    1st: path to the directory that stores the DICOM files. The expectation 
        here is that the files have already been "cleaned" and only files
        associated with the patient study and relevant for processing exist
        in the directory.
    2nd: path to desired output file location. Note that your file should end
        with the suffix ".h5" since the file is saved as HDF5 format.
        
    NAMED
    -px, --pixel_size: float value to set what size to rescale pixels to.
            Note that the script does not modify slice thickness, so you don't
            have total control over voxel size.
    -pl, --parotid_l: just a boolean flag, tells script to gather mask data
            for left parotid
    -pr, --parotid_r: same as -pl but for right parotid. if both are included,
            the script will create a merged mask of both parotids
    -bb, --bounding_box: 3D axes presecription for  output array size. Note that
            arrays are generated as Z,Y,X axis order.
    -c, --center_of_mass: boolean flag telling the program whether to center the
            bounding box on the mask center of mass.
    -a, --augments: integer, number of augments to generate. currently only
            supports random augments.
    -l, --label: path to label file
    -ch, --pt_chars: path to one-hot-encoded patient characteristics file


"""



def main():
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
    parser.add_argument(
        '-a',
        '--augments',
        type=int,
        default=0,
        help='Sets the number of data augments to also generate'
        )
    parser.add_argument(
        '-l','--label',type=str,default=None,help="Path to label file."
        )
    parser.add_argument(
        '-ch',
        '--pt_chars',
        type=str,
        default=None,
        help="Path to OHE patient characteristic file."
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
    
    if args.label is not None:
        labeldf = pd.read_csv(args.label,index_col=0)
        prepper.get_label(labeldf)
        
    if args.pt_chars is not None:
        pc_file = pd.read_csv(args.pt_chars,index_col=0)
        prepper.get_pt_chars(pc_file)
    
    prepper.save(
        args.destination,
        boxed=boxed,
        boxshape=boxsize,
        maskcentered=args.center_of_mass
        )
    
    if args.augments > 0:
        for i in range(args.augments):
            prepper.random_augment()
            prepper.save(
                args.destination,
                boxed=boxed,
                boxshape=boxsize,
                maskcentered=args.center_of_mass
                )
            prepper.reset_augments()
            
if __name__ == "__main__":
    main()