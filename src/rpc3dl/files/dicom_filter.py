# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 19:52:44 2023

@author: johna

Defines the Filter class, which is used to process the bulk export of DICOM
files from MIM into local storage.
"""

import os
import shutil
import pydicom

import rpc3dl.files._dicom_util as dcmutil

from contextlib import redirect_stdout

class Filter:
    def __init__(self,
                 keep=['CT','RTPLAN','RTDOSE','RTSTRUCT']):
        self.keep_modality = keep
        
    def set_endpoints(self,source,dest):
        self.source = source
        self.dest = dest
        
    def send_files(self,modality_filter=True):
        """
        Function to send files from source endpoint to dest endpoint.
        Assumes that source endpoint contains all patient files.
        
        If modality_filter is True, then it only sends files that fit the
        modality.
        """
        filenames = os.listdir(self.source)
        filepaths = [os.path.join(self.source,file) for file in filenames]
        filepaths = [path for path in filepaths if os.path.isfile(path)]
        # this setup does not walk into any subdirs
        
        if not os.path.exists(self.dest):
            os.mkdir(self.dest)
        
        for path in filepaths:
            if modality_filter:
                file = pydicom.dcmread(path)
                if file.Modality not in self.keep_modality:
                    continue
            shutil.move(path,path.replace(self.source,self.dest))
            
    def sort_studies(self,patientfolder=None):
        # meant to be called after send_files
        # performs per-study subdir organization on dest endpoint
        
        if patientfolder is None:
            patientfolder = self.dest
        filenames = os.listdir(patientfolder)
        for file in filenames:
            path = os.path.join(patientfolder,file)
            try:
                tempfile = pydicom.dcmread(path)
            except pydicom.errors.InvalidDicomError:
                continue
            studyUID = tempfile.StudyInstanceUID
            studydir = os.path.join(patientfolder,studyUID)
            if not os.path.exists(studydir):
                os.mkdir(studydir)
            shutil.move(path,os.path.join(studydir,file))
            
    def find_planning_study(self,
                            location=None,
                            cleanup=True,
                            probe=False):
        if location is None:
            patientfolder = self.dest
        else:
            patientfolder = location
            
        allfiles = []
        for file in os.listdir(patientfolder):
            path = os.path.join(patientfolder,file)
            allfiles.append(path)
        planningstudy = dcmutil.get_planning_study(allfiles)
        if planningstudy is None:
            print(
                "Strict planning study search failed, trying loose search"
                )
            planningstudy = dcmutil.find_complete_study(allfiles)
            if planningstudy is None:
                print("Loose search failed")
                return None
        if probe is True:
            # early exit, for if you want to examine the state of data without
            # affecting the files at all
            return True
        
        destination = os.path.join(patientfolder,"data")
        os.mkdir(destination)
        naming = {"CT":"CT","RTSTRUCT":"RS","RTDOSE":"RD","RTPLAN":"RP"}
        for i, dcm in enumerate(planningstudy):
            if str(dcm.Modality) not in naming.keys():
                naming[str(dcm.Modality)] = str(dcm.Modality)
            filename = naming[str(dcm.Modality)] + "_{}.dcm".format(i)
            pydicom.write_file(os.path.join(destination,filename), dcm)
        
        if cleanup is True:
            for file in allfiles:
                os.remove(file)
        
if __name__ == "__main__":
    test = Filter()
    sourcedir = r"D:\H_N"
    with open("log.txt","w") as f:
        with redirect_stdout(f):
            for patient in os.listdir(sourcedir):
                subdirpath = os.path.join(sourcedir,patient)
                if not os.path.isdir(subdirpath):
                    continue
                try:
                    check = test.find_planning_study(subdirpath,debug=True)
                except Exception as e:
                    print(e)
                    check = None
                if check is not None:
                    print("Done with {} - success".format(patient))
                else:
                    print("Done with {} - failure".format(patient))
        f.close()