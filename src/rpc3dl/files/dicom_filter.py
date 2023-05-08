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

"""
Dose files reference plan files:
    dosefile.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID
Plan files reference structure set files:
    plan.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID
Structure Set files reference CT series:
    ssfile.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID
    (note that this value maps to SERIES instance UID, not an SOPInstanceUID)

"""

class Filter:
    def __init__(self,
                 keep=['CT','RTPLAN','RTDOSE','RTSTRUCT']):
        self.keep_modality = keep
        self.support_modalities = []
        self.patientID = None
        
    def set_endpoints(self,source,dest,support=None):
        self.source = source
        self.dest = dest
        self.support = support
        if support is not None:
            for mod in ['RTPLAN','RTDOSE','RTSTRUCT']:
                self.keep_modality.remove(mod)
                self.support_modalities.append(mod)
        
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
            file = pydicom.dcmread(path)
            if file.PatientID != self.patientID:
                if self.patientID is not None:
                    print("Patient ID mismatch:",file.PatientID, self.patientID)
            self.patientID = file.PatientID
            if modality_filter:
                if file.Modality not in self.keep_modality:
                    continue
            shutil.move(path,path.replace(self.source,self.dest))
            
    def fetch_supportfiles(self):
        for root, dirs, files in os.walk(self.support):
    
            for f in files:
                filepath = os.path.join(root,f)
                try:
                    temp = pydicom.dcmread(filepath)
                except:
                    continue
                if temp.PatientID in self.patientID:
                    shutil.move(
                        filepath,
                        os.path.join(self.dest,os.path.basename(filepath))
                        )
            
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
            
    def filter_files(self,
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
        cts, plan, dose, ss = dcmutil.walk_references(allfiles)
        if any([cts is None, dose is None, ss is None]):
            raise Exception("Missing critical files")
        if probe is True:
            # early exit, for if you want to examine the state of data without
            # affecting the files at all
            return True
        
        alldcms = cts # this will always be a list
        for f in [plan, dose, ss]:
            if f is not None:
                alldcms.append(f)
        frame_of_ref = []
        for dcm in alldcms:
            frameUID = dcmutil.get_attr_deep(dcm, "FrameOfReferneceUID")
            if frameUID not in frame_of_ref:
                frame_of_ref.append(frameUID)
        if len(frame_of_ref) > 1:
            print("Warning: More than one Frame of Reference in files.")
        
        os.mkdir(os.path.join(location,"temp"))
        for file in allfiles:
            shutil.move(
                file,
                file.replace(location,os.path.join(location,"temp"))
                )
        
        naming = {"CT":"CT","RTSTRUCT":"RS","RTDOSE":"RD","RTPLAN":"RP"}
        for i, dcm in enumerate(alldcms):
            if str(dcm.Modality) not in naming.keys():
                naming[str(dcm.Modality)] = str(dcm.Modality)
            filename = naming[str(dcm.Modality)] + "_{}.dcm".format(i)
            pydicom.write_file(os.path.join(location,filename), dcm)
        
        if cleanup is True:
            shutil.rmtree(os.path.join(location,"temp"))
        
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