# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 22:53:26 2023

@author: johna
"""

import os

import numpy as np
import pydicom
from pydicom.sequence import Sequence
from pydicom.errors import InvalidDicomError

"""
Main need: function that can iterate through DICOM files to determine which
should be set aside for use.

For my main research, I need the planning CT study, dose file, and associated
structure set file (which contains, at minimum, both parotid glands). However,
there may be use in the future in allowing customization of parameters to set
what the requirements of file acceptance are.
"""

def is_dicom_file(filename):
    with open(filename, 'rb') as f:
        header = f.read(4)
        if header == b'DICM':
            return True
        else:
            return False

def organize_list(dcmlist,attribute):
    org_dict = {}
    for dcm in dcmlist:
        val = getattr(dcm,attribute)
        if val not in org_dict.keys():
            org_dict[val] = []
        org_dict[val].append(dcm)
    return org_dict

def same_FoR(dcms):
    """
    Quick check to confirm that all DICOM files in a list of DICOM files share
    the same frame of reference.
    
    Intent is to use it on a full study. Accepts list or modality-sorted dict.
    """
    if isinstance(dcms, dict):
        to_check = []
        for modality in dcms.keys():
            to_check += dcms[modality]
    elif isinstance(dcms,list):
        to_check = dcms
    
    FoR = []
    for file in to_check:
        FoR.append(get_attr_deep(file,"FrameOfReferenceUID"))
    check = set(FoR)
    return len(check) == 1
        
        
def get_attr_deep(dcm,searchfor):
    """
    Deep attribute fetcher that can also retrieve a DICOM attribute out of the 
    first layer of a Sequence. This is necessary tool for FrameOfReferenceUID
    as the structure set files tuck it into the first layer of the
    Referenced Frame of Reference Sequence.
    
    This function, however, is written to be generically usable.
    
    Parameters
    ----------
    dcm : pydicom FileDataset
        DICOM file to get attribute from
    searchfor : str
        Attribute to retrieve
    
    Returns
    -------
    value
        Attribute requested from the DICOM file.
    """
    value = None
    if hasattr(dcm,searchfor):
        value = getattr(dcm,searchfor)
    else:
        for attribute in dir(dcm):
            if not attribute[0].isupper():
                # Pydicom sets DICOM attributes in camelcase
                continue
            if isinstance(getattr(dcm,attribute),Sequence):
                for item in getattr(dcm,attribute):
                    if hasattr(item, searchfor):
                        value = getattr(item,searchfor)
                        break
    return value

def levenshteinDistanceDP(token1, token2):
    # Source: 
    # https://blog.paperspace.com/
    # implementing-levenshtein-distance-word-autocomplete-autocorrect/
    
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2
        
    a = 0
    b = 0
    c = 0
    
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1-1] == token2[t2-1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]
                
                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]

def parotid_check(ss):
    l_common_names = ['parotid l','parotid (left)', 'left parotid']
    r_common_names = ['parotid r','parotid (right)', 'right parotid']
    checknames = [
        ROI.ROIName.lower().replace("_"," ") \
            for ROI in ss.StructureSetROISequence
        ]
    leftcheck = any([name in l_common_names for name in checknames])
    rightcheck = any([name in r_common_names for name in checknames])
    return (leftcheck and rightcheck)

def hierarchy(list_of_dcms,level='patient'):
    """ Hierarchical organization of files as follows:
        PatientID
         |__StudyInstanceUID
             |__Modality
             
    To be used after loading a bulk batch of files into a list to sort them.
    
    level parameter sets what level to return. If patient, default behavior.
    If study, steps down to make study top level
    """
    patientsort = organize_list(list_of_dcms,"PatientID")
    for patient_id, patient_files in patientsort.items():
        patientsort[patient_id] = organize_list(
            patient_files,"StudyInstanceUID"
            )
        for study, study_files in patientsort[patient_id].items():
            patientsort[patient_id][study] = organize_list(
                study_files, "Modality"
                )
    # Hierarchical data organization complete
    
    # Step down as needed based on requested organization level
    if level.lower() in ["study","modality"]:
        patientsort = _step_down_hierarchy(patientsort)
        if level.lower() == "modality":
            patientsort = _step_down_hierarchy(patientsort)
    
    return patientsort

def _step_down_hierarchy(hier):
    """Function to step one level down in the hierarchy. Note that this
    will drop the current top level of the hierarchy.
    
    Useful if files are already organized by PatientID and you just need to
    process at study-level.
    """
    new = {}
    for subdict in hier.values():
        for k,v in subdict.items():
            new[k] = v
    return new

def validate_study(study_dict,oar_check_method=parotid_check):
    # must have files for all three essential modalities
    for modality in ['CT','RTDOSE','RTSTRUCT']:
        if modality not in study_dict.keys():
            print("Missing critical modality:",modality)
            return False
    # files must all share a FrameOfReferenceUID
    #if not same_FoR(study_dict):
    #    print("Too many frames of reference in study.")
    #    return False
    # check for whether the structure set file has what we need
    keep_ss = [ss for ss in study_dict['RTSTRUCT'] if oar_check_method(ss)]
    #if len(keep_ss) != 1:
    #    print(
    #        "Bad number of ss files accepted: {}".format(len(keep_ss))
    #        )
    #    return False
    # replace ss list with only valid ss
    # TODO - double check this behavior, unsure if editing study_dict is ok
    study_dict['RTSTRUCT'] = keep_ss
    return True

def get_planning_study(filepaths):
    status = ""
    dcms = [pydicom.dcmread(file) for file in filepaths]
    modality_dict = hierarchy(dcms,level="modality")
    for m in ['CT','RTPLAN','RTDOSE','RTSTRUCT']:
        if m not in modality_dict.keys():
            status += "Missing {} | ".format(m)
    
    if status != "":
        print(status)
        return None
    # first validate plan files
    approved_plans = []
    for file in modality_dict['RTPLAN']:
        if str(file.ApprovalStatus).upper() == "APPROVED":
            approved_plans.append(file)
            
    # if we don't find just one approved plan, abort filtering
    num_approved_plans = len(approved_plans)
    status += "{} approved plans found | ".format(num_approved_plans)
    if num_approved_plans != 1:
        print(status)
        return None
    else:
        modality_dict['RTPLAN'] = approved_plans
        plan = modality_dict['RTPLAN'][0]
        ss_ref_uid = plan.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID
        plan_ref_uid = plan.SOPInstanceUID
    
    paired_dose = []
    for dose in modality_dict['RTDOSE']:
        refplan = dose.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID
        if refplan == plan_ref_uid:
            # ensure dose object refers to approved plan object
            paired_dose.append(dose)
    status += "{} dose files found that match plan reference | ".format(
        len(paired_dose)
        )
    if len(paired_dose) < 1:
        print(status)
        return None
    else:
        modality_dict['RTDOSE'] = paired_dose
        
    correct_ss = []    
    for ss in modality_dict['RTSTRUCT']:
        if ss.SOPInstanceUID == ss_ref_uid:
            # ensure ss object is correctly tied to plan object
            correct_ss.append(ss)
    status += "{} structure sets found that match UID reference | ".format(
        len(correct_ss)
        )
    if len(correct_ss) != 1:
        print(status)
        return None
    else:
        modality_dict['RTSTRUCT'] = correct_ss
    
    ss = modality_dict['RTSTRUCT'][0]
    refseq = ss.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence
    instanceuids = []
    for ea in refseq:
        instanceuids.append(ea.ReferencedSOPInstanceUID)
        
    ct = []
    for img in modality_dict['CT']:
        if img.SOPInstanceUID in instanceuids:
            # only accept CT images associated with structure set object
            ct.append(img)
    modality_dict['CT'] = ct
    status += "{} CT files associated with structure set | ".format(len(ct))
    
    dcmfiles = []
    for modalityfiles in modality_dict.values():
        for ea in modalityfiles:
            dcmfiles.append(ea)
    print(status)
    return dcmfiles

def find_complete_study(dcmlist):
    # fallback function for if strict planning study filter fails
    dcmlist = [pydicom.dcmread(file) for file in dcmlist]
    study_dict = hierarchy(dcmlist, level='study')
    studies = list(study_dict.keys())
    good = []
    for study in studies:
        if all([m in study_dict[study].keys()
                for m in ['CT','RTPLAN','RTDOSE','RTSTRUCT']]):
            good.append(study)
    if len(good) == 1:
        goodstudy = study_dict[good[0]]
        files = []
        for k,v in goodstudy.items():
            for file in v:
                files.append(file)
        return files
    else:
        return None
        

def main_filter(source_folder,
                destination_folder,
                modalitykeep=['CT','RTSTRUCT','RTDOSE']):
    """ Assumes source_folder contains DICOM files of a single patientID,
    checks studies, saves valid studies to subfolders in destination folder.
    """
    files = os.listdir(source_folder)
    success = False
    dcms = []
    goodstudy = []
    # loop to load all DICOM files
    for file in files:
        try:
            dcm = pydicom.dcmread(os.path.join(source_folder,file))
        except InvalidDicomError:
            continue # skip anything that's not dicom
        dcms.append(dcm)
    hier = hierarchy(dcms,level='study') # organize files in hierarchy
    print("Number of studies: {}".format(len(list(hier.keys()))))
    for i, (study, studydict) in enumerate(hier.items()):
        if validate_study(studydict):
            goodstudy.append(study)
            success = True
            os.mkdir(os.path.join(destination_folder,str(study)))
            
    # studies all evaluated at this point, now we move files over
    #
    for file in files:
        try:
            dcm = pydicom.dcmread(os.path.join(source_folder,file))
        except InvalidDicomError:
            continue # skip anything that's not dicom
        if dcm.StudyInstanceUID in goodstudy and dcm.Modality in modalitykeep:
            destfullpath = os.path.join(destination_folder,
                                        str(dcm.StudyInstanceUID),
                                        file)
            pydicom.write_file(destfullpath,dcm)
    return success