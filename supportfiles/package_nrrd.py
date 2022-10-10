# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 21:43:58 2022

@author: johna
"""

import numpy as np
import pydicom
import pandas as pd
import nrrd

import os

import data_utils as util

# ==== Files necessary to run program ====
SOURCE_DIR = "D:\\H_N"
SOURCE_OVERRIDE = "F:\\DICOMdata\\RoswellData"
DEST_DIR = "D:\\xero_nrrd\\dose_scale"
GUIDE_FILE = "goodtogo.csv"
GUIDE_OVERRIDE = "goodtogo2.csv"
SURVEY_FILE = "D:\\H_N\\HeadAndNeckQOLSurvey-General_anon_with_time_since.csv"

# ==== Settings ====
SCALETO = 'dose'


guide = pd.read_csv(GUIDE_FILE)
guide_override = pd.read_csv(GUIDE_OVERRIDE)
surveys = pd.read_csv(SURVEY_FILE)

patients_core = list(guide['patient'].values)
patients_override = list(guide_override['patient'].values)

patients = np.unique(np.array(patients_core + patients_override))

for patient in patients:    
    if patient == '018_004':
        continue
    
    # ======= BEGIN VALIDATION BLOCK =========
    if os.path.exists(os.path.join(DEST_DIR,patient+'.nrrd')):
        # perform validation tests
        print(patient,"already processed, performing validations...")
        filepath = os.path.join(DEST_DIR,patient+'.nrrd')
        tmpdata, tmpheader = nrrd.read(filepath)
        rewrite_data = False
        # check labels
        if any(('early_label' not in tmpheader.keys(),
                'late_label' not in tmpheader.keys())):
            earlylabel, latelabel = util.assign_labels(patient,surveys)
            tmpheader['early_label'] = earlylabel
            tmpheader['late_label'] = latelabel
            rewrite_data = True
            
        # check source
        if 'source_path' not in tmpheader.keys():
            if patient in patients_override:
                location = os.path.join(SOURCE_OVERRIDE,patient)
            else:
                location = os.path.join(SOURCE_DIR,patient)
            tmpheader['source_path'] = location
            rewrite_data = True
        
        # get pixel spacing
        if 'pixel_spacing' not in tmpheader.keys():
            if tmpheader['scaleto'] == 'img':
                for file in os.listdir(tmpheader['source_path']):
                    if file.startswith("CT") and file.endswith(".dcm"):
                        tempfile = pydicom.read_file(os.path.join(tmpheader['source_path'],file))
                        pixel_spacing = tempfile.PixelSpacing
                        tmpheader['pixel_spacing'] = pixel_spacing
                        break
            elif tmpheader['scaleto'] == 'dose':
                for file in os.listdir(tmpheader['source_path']):
                    if file.startswith("RD") and file.endswith(".dcm"):
                        tempfile = pydicom.read_file(os.path.join(tmpheader['source_path'],file))
                        pixel_spacing = tempfile.PixelSpacing
                        tmpheader['pixel_spacing'] = pixel_spacing  
            rewrite_data = True
        
        if rewrite_data == True:
            nrrd.write(filepath,tmpdata,header=tmpheader)
        continue

    # ========== END VALIDATION PORTION ==========
    # Begin creation of new nrrd

    stdheader = {'axismap':['z (slices)','y (rows)','x (columns)','channels'],
                 'channelmap':['image','dose','parotid mask'],'scaleto':SCALETO}
    
    # check source
    if patient in patients_override:
        location = os.path.join(SOURCE_OVERRIDE,patient)
    else:
        location = os.path.join(SOURCE_DIR,patient)
    stdheader['source_path'] = location

    dosefiles = []
    ssfiles = []
    imgfiles = []
    
    patientdir = location
    
    files = os.listdir(patientdir)
    
    # =====================
    # Load all relevant DICOM files based on filename
    # =====================
    
    for file in files:
        if not file.endswith('.dcm'):
            continue
        if 'IGNORE' in file:
            continue
        if file.startswith("CT"):
            imgfiles.append(pydicom.read_file(os.path.join(patientdir,file)))
        elif file.startswith("RD"):
            dosefiles.append(pydicom.read_file(os.path.join(patientdir,file)))
        elif file.startswith("RS"):
            ssfiles.append(pydicom.read_file(os.path.join(patientdir,file)))
            
    # ====================
    # Validate inputs (number of dose/ss files, consistency of img UIDs)
    # ====================
    if len(dosefiles) > 1:
        try:
            mergeddose = util.merge_doses(*dosefiles)
        except ValueError:
            print("Dose merge failed for {}, bypassing...".format(patient))
            continue
        dose = dosefiles[0]
        dose.PixelData = mergeddose.tobytes() # replace dose values with merged
    
    else:
        dose = dosefiles[0]
    if len(ssfiles) > 1:
        for i, tempss in enumerate(ssfiles):
            if len(util.list_contours(tempss)) == 1:
                del ssfiles[i]
        if len(ssfiles) > 1:
            raise Exception("Too many ss files for" +
                            " {}, please mark one for ignore".format(patient))
    
    imgUIDs = []
    for img in imgfiles:
        imgUIDs.append(img.SeriesInstanceUID)
    uniqueUIDs = list(set(imgUIDs))
    if len(uniqueUIDs) > 1:
        stdheader['warning'] = "Potential UID inconsistencies in images."
        
    ss = ssfiles[0]
    dose = dosefiles[0]
    
    if stdheader['scaleto'] == 'dose':
        stdheader['pixel_spacing'] = dose.PixelSpacing
    
    # ===================
    # Create dictionary mapping 2D, 3-channel arrays to slice heights
    # ===================    
    mapping = {}
    
    # --- Need to programmatically extract parotid contours -----
    # note that this method is not necessarily foolproof, we'll want to 
    # look at improving at some point
    all_contours = util.list_contours(ss)
    parotids = {k:v for k,v in all_contours.items() if 'parotid' in k.lower()}
    parotids = {k:v for k,v in parotids.items() if 'stem' not in k.lower()}
    r_parotid = None
    l_parotid = None
    for k,v in parotids.items():
        if any(("l" in x for x in k.lower().split('parotid'))):
            try:
                l_parotid = util.get_contour(ss,v)
            except AttributeError:
                continue
        elif any(("r" in x for x in k.lower().split('parotid'))):
            try:
                r_parotid = util.get_contour(ss,v)
            except AttributeError:
                continue
            
    slicethicknesses = []
    for img in imgfiles:
        # validate pixel spacing, record in stdheader if img is scaleto
        pixel_spacing = img.PixelSpacing
        if stdheader['scaleto'] == 'img':
            if 'pixel_spacing' not in stdheader.keys():
                stdheader['pixel_spacing'] = pixel_spacing
            else:
                if any((pixel_spacing[0] != stdheader['pixel_spacing'][0],
                        pixel_spacing[1] != stdheader['pixel_spacing'][1])):
                    print("Mismatched pixel spacing present in patient files")
                    raise Exception("Non-conforming pixel spacings.")
        z = float(img.SliceLocation)
        slicethicknesses.append(img.SliceThickness)
        if l_parotid is None:
            lp_mask = np.zeros_like(img.pixel_array)
        else:
            lp_coords = util.pull_single_slice(l_parotid,img)
            lp_mask = util.coords_to_mask(lp_coords,img)
        
        if r_parotid is None:
            rp_mask = np.zeros_like(img.pixel_array)
        else:
            rp_coords = util.pull_single_slice(r_parotid,img)
            rp_mask = util.coords_to_mask(rp_coords,img)
        unified_mask = lp_mask + rp_mask
        if np.any((unified_mask != 1)&(unified_mask != 0)):
            raise Exception("Something went wrong with mask generation")
            
        finalimg, finaldose, finalmask = util.get_slices(
            img,dose,unified_mask,scaleto=SCALETO
            )
        if all((x is None for x in [finalimg,finaldose,finalmask])):
            continue
        mapping[z] = np.array([finalimg,finaldose,finalmask])
    
    sortedmapping = dict(sorted(mapping.items()))
    final4Darray = np.array([x for x in sortedmapping.values()])
    final4Darray = np.moveaxis(final4Darray,1,-1)
    
    stdheader['slicemap'] = list(sortedmapping.keys())
    uniquethicknesses = list(set(slicethicknesses))
    if len(uniquethicknesses) > 1:
        print("{p} has more than one slice thickness, check mapping".format(
            p=patient
            ))
    stdheader['slicethicknesses'] = uniquethicknesses
    liveboxshape = util.measure_box(final4Darray)
    stdheader['liveboxshape'] = liveboxshape
    
    earlylabel, latelabel = util.assign_labels(patient,surveys)
    stdheader['early_label'] = earlylabel
    stdheader['late_label'] = latelabel
    
    nrrd.write(os.path.join(DEST_DIR,patient+'.nrrd'),final4Darray,header=stdheader)
    print("Completed {}".format(patient))