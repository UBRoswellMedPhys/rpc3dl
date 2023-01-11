# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 09:54:20 2022

@author: johna
"""

import os
import pydicom
import numpy as np
import pandas as pd
import string
import random

import lookups # will need to change this to a relative import eventually
import dummy

from pydicom.tag import Tag
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError

profile = lookups.basic_profile
UID_map = {} # will be filled
anonID_map = {}

def get_iod(dcm):
    modality = dcm.Modality
    if modality == "CT":
        iod = lookups.ct_image_iod
    elif modality == "MR":
        iod = lookups.mr_image_iod
    elif modality == "PET":
        iod = lookups.pet_image_iod
    elif modality == "RTSTRUCT":
        iod = lookups.rt_structure_set_iod
    elif modality == "RTDOSE":
        iod = lookups.rt_dose_iod
    elif modality == "RTPLAN":
        iod = lookups.rt_plan_iod
    else:
        raise InvalidDicomError(
            "DICOM type {} not yet supported".format(modality)
            )
    return iod

def deident_seq():
    # this is for Tag(('0012','0064'))
    seq = Dataset()
    seq.CodeValue = '113100'
    seq.CodingSchemeDesignator = 'DCM'
    seq.CodeMeaning = 'Basic Application Confidentionality Profile'
    return seq

def deident_info(dcm):
    dcm[Tag(('0012','0062'))].value = "YES" # Patient Identity Removed
    dcm[Tag(('0012','0063'))].value = "De-identified with pydicom package"
    dcm[Tag(('0012','0064'))].value = deident_seq()

dummy_VR = {
    'CS': dummy.dummy_CS,
    'SH': dummy.dummy_string,
    'LO': dummy.dummy_string,
    'ST': dummy.dummy_string,
    'LT': dummy.dummy_string,
    'UT': dummy.dummy_string,
    'PN': dummy.dummy_string,
    'UI': dummy.dummy_UID,
    'DA': dummy.dummy_date,
    'TM': dummy.dummy_time,
    'DT': dummy.dummy_datetime,
    'AS': dummy.dummy_age,
    'IS': dummy.dummy_int,
    'DS': dummy.dummy_decimal,
    'OW': dummy.dummy_string,
    'OF': dummy.dummy_string
    }

def assign_anon_id(dcm,anonid):
    dcm[Tag(('0010','0020'))].value = anonid

def anonymize_element(dataset, element, profile, iod):
    global UID_map
    if any(('instance uid' in element.description().lower(),
            element.tag == Tag(('0020','0052')))):
        olduid = element.value
        if olduid in UID_map:
            element.value = UID_map[olduid]
        else:
            UID_map[olduid] = dummy.dummy_UID()
            element.value = UID_map[olduid]
            
    if element.tag in profile.keys():
        procedure = profile[element.tag]
        
        if len(procedure) > 1:
            # variable procedure code, push through IOD checks
            if iod is None:
                raise Exception("Need a defined IOD")
            options = []
            for module in iod:
                if element.tag in module.keys():
                    options.append(module[element.tag])
            for code in ['X','Z','D','U']:
                if code in options:
                    procedure = code 
                    # prioritizes D over Z and Z over X by iteration order
        if len(procedure) > 1:            
            # if IOD checker does not resolve variable procedure code then
            # take the most conservative method (last in sequence)
            procedure = procedure[-1]
        
        
        if procedure == 'X':
            del dataset[element.tag]
        elif procedure == 'Z':
            element.value = ''
        elif procedure == 'D':
            element.value = dummy_VR[element.VR]()
        elif procedure == 'U':
            # UID handler is performed above - all instance UIDs are being 
            # masked to dummy UIDs, regardless of presence in Basic Profile
            pass
        else:
            print(
                "Failed to find acceptable procedure code ({})".format(procedure)
                )
            
def iod_wrapper(profile,iod):
    def anon_callback(dataset,element,profile=profile,iod=iod):
        anonymize_element(dataset,element,profile,iod)
    return anon_callback

def anonymize_file(file):
    global anonID_map
    oldID = file.PatientID
    if oldID not in anonID_map.keys():
        newID = dummy.dummy_patientID()
        anonID_map[oldID] = newID
    else:
        newID = anonID_map[oldID]
    iod = get_iod(file)
    callback = iod_wrapper(profile,iod)
    file.walk(callback)
    assign_anon_id(file,newID)
    return file
    
def save_file(dcm,dest):
    basefilename = "{m}.{i}".format(m=dcm.Modality,i=dcm.PatientID)
    i = 1
    while True:
        filename = "{f}.{n}.dcm".format(f=basefilename,n=i)
        filepath = os.path.join(dest,filename)
        if not os.path.exists(filepath):
            pydicom.write_file(filepath,dcm)
            break
        i += 1

def process_csvs(csvs, args):
    """
    Function to handle anonymization of CSV format data. Depends on already
    having processed all DICOM files, as they contribute to the global
    variable of anonID map, which is used to keep ID matching consistent
    in anonymization.
    
    Parameters
    ----------
    csvs : dict
        Dictionary of filepath : pd.DataFrame pairs.
    args : argparser read of CLI arguments
    """
    processedIDs = np.array(list(anonID_map.keys())) # anonID_maps global var
    for path, df in csvs.items():
        # check to see if any of the processed IDs occur in the CSV
        if not df.isin(processedIDs).any(axis=None):
            # if no processed IDs (old IDs) exist in cells in the CSV, continue
            print(
                "No data in {}, continuing...".format(os.path.split(path)[1])
                )    
            continue
        else:
            # this means IDs have been found - first, reduce irrelevant rows
            liverows = df.isin(processedIDs).any(axis=1)
            df = df.iloc[liverows[liverows].index]
            # now only retained rows with IDs that are relevant
            cols_contain_IDs = df.isin(processedIDs).any(axis=0)
            for col in cols_contain_IDs[cols_contain_IDs].index:
                # goes through columns that contain IDs and remaps them
                df[col] = df[col].apply(lambda x: anonID_map[x])
        destination = path.replace(args.path, args.dest)
        if os.path.exists(destination):
            existing_df = pd.read_csv(destination)
            merged = pd.concat([existing_df, df], axis=0)
            merged.to_csv(destination,index=False)
        else:
            df.to_csv(destination,index=False)
        
def cleanpath(path):
    # pulls ID map from global to make sure the filepath does not contain
    # patient IDs - if it does, anonymizes those as well
    for ptID in anonID_map.keys():
        if ptID in path:
            path = path.replace(ptID, anonID_map[ptID])
    return path

def main(args):
    # TODO - break this up into subfunctions as necessary, too bulky right now
    destroot = cleanpath(args.dest)
    # if destination directory does not exist, create it
    if not os.path.exists(destroot):
        os.mkdir(destroot)
    # check whether the path points to a file or a directory
    if os.path.isfile(args.path):
        try:
            dcm = pydicom.dcmread(args.path)
            get_iod(dcm) # validates modality
        except InvalidDicomError:
            print(
                "Filepath provided does not point to a valid DICOM file."
                )
            raise
        dcm = anonymize_file(dcm)
        save_file(dcm,destroot)
        
    elif os.path.isdir(args.path):
        # if path points to a directory, cycle through all contents
        csvs = {}
        if not args.recursive:
            for file in os.listdir(args.path):
                filepath = os.path.join(args.path, file)
                if not os.path.isfile(filepath):
                    continue # skip any subdirectories
                if filepath.endswith(".csv"):
                    csvs[filepath] = pd.read_csv(filepath)
                    continue # stash to process at the end
                try:
                    dcm = pydicom.dcmread(filepath)
                    get_iod(dcm) # validates modality
                except InvalidDicomError:
                    print(
                        "Cannot process {}, bypassing.".format(file)
                        )
                    continue
                dcm = anonymize_file(dcm)
                save_file(dcm, destroot)
                
        elif args.recursive:
            for root, subdirs, files in os.walk(args.path, topdown=True):
                destfolder = root.replace(args.path,args.dest)
                destfolder = cleanpath(destfolder)
                for file in files:
                    filepath = os.path.join(root,file)
                    if filepath.endswith(".csv"):
                        csvs[filepath] = pd.read_csv(filepath)
                        continue # stash to process at the end
                    try:
                        dcm = pydicom.dcmread(filepath)
                        get_iod(dcm) # validates modality
                    except InvalidDicomError:
                        print(
                            "Cannot process {}, bypassing.".format(file)
                            )
                        continue
                    dcm = anonymize_file(dcm)
                    destfolder = cleanpath(destfolder)
                    if not os.path.exists(destfolder):
                        os.mkdir(destfolder)
                    save_file(dcm, destfolder)
                # once all files are processed, do a last "does it exist"
                # check in case there are directories with no files that
                # need to be copied over to retain folder structure
                destfolder = cleanpath(destfolder)
                if not os.path.exists(destfolder):
                    os.mkdir(destfolder)
        
        process_csvs(csvs, args)
        
    else:
        raise Exception("Invalid target path provided.")  

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog="DICOMAnon",
        description="Program to anonymize DICOM files for DL research."
        )
    parser.add_argument(
        "-p","--path",
        dest="path",
        required=True,
        help="Path to file or directory containing DICOM files to anonymize."
        )
    parser.add_argument(
        '-d','--dest',
        dest="dest",
        required=True,
        help="Path to destination folder to store anonymized files."
        )
    parser.add_argument(
        "-m","--map",dest="save_map",action="store_true",
        help="Save mapping to CSV"
        )
    parser.add_argument(
        "-r","--recursive", dest="recursive", action="store_true", 
        default=False, help="Recursive processing of target directory."
        )
    args = parser.parse_args()
    
    main(args)