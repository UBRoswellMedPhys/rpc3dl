# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 09:54:20 2022

@author: johna
"""

import os
import pydicom
import pandas as pd
import string
import random

import lookups # will need to change this to a relative import eventually

from pydicom.tag import Tag
from pydicom.dataset import Dataset

profile = lookups.basic_profile
UID_map = {} # will be filled

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
        raise Exception("DICOM type {} not yet supported".format(modality))
    return iod

def deident_seq():
    seq = Dataset()
    seq.CodeValue = '113100'
    seq.CodingSchemeDesignator = 'DCM'
    seq.CodeMeaning = 'Basic Application Confidentionality Profile'
    return seq

def dummy_data(size=8, chars=string.ascii_letters + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def dummy_CS():
    chars = string.ascii_uppercase + string.digits + '_'
    return dummy_data(chars=chars)

def dummy_string():
    chars = string.ascii_letters + string.digits
    return dummy_data(chars=chars)

def UID_handler():
    raise Exception("Not built yet")
    # TODO
    return None

def dummy_date():
    return '19000101'

def dummy_time():
    return '123000.00'

def dummy_datetime():
    return dummy_date() + dummy_time()

def dummy_age():
    return '099Y'

def dummy_int():
    return dummy_data(size=6,chars=string.digits)

def dummy_decimal():
    return dummy_data(size=5,chars=string.digits) + ".0"


dummy_VR = {
    'CS': dummy_CS,
    'SH': dummy_string,
    'LO': dummy_string,
    'ST': dummy_string,
    'LT': dummy_string,
    'UT': dummy_string,
    'PN': dummy_string,
    'UI': UID_handler,
    'DA': dummy_date,
    'TM': dummy_time,
    'DT': dummy_datetime,
    'AS': dummy_age,
    'IS': dummy_int,
    'DS': dummy_decimal,
    'OW': dummy_string,
    'OF': dummy_string
    }

def assign_anon_id(dcm,anonid):
    dcm[Tag(('0010','0020'))].value = anonid
    return dcm

def anonymize_element(dataset, element, profile, iod):
    global UID_map
    if 'instance uid' in element.description().lower():
        olduid = element.value
        if olduid in UID_map:
            element.value = UID_map[olduid]
        else:
            UID_map[olduid] = dummy_UID()
            element.value = UID_map[olduid]
            
    if element.tag in profile.keys():
        procedure = profile[element.tag]
        
        if procedure not in ['X','Z','D','U']:
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
                    #prioritizes D over Z and Z over X by iteration order
        
        if procedure == 'X':
            del dataset[element.tag]
        elif procedure == 'Z':
            element.value = ''
        elif procedure == 'D':
            element.value = dummy_VR[element.VR]()
        elif procedure == 'U':
            # TODO - build UID handlers
            pass
        else:
            raise Exception(
                "Failed to find acceptable procedure code ({})".format(procedure)
                )
            
def iod_wrapper(profile,iod):
    def anon_callback(dataset,element,profile=profile,iod=iod):
        anonymize_element(dataset,element,profile,iod)
    return anon_callback

def dummy_UID():
    UID = "1.2.246.352.221." + random.choice("123456789")
    while len(UID) < 59:
        UID += random.choice(string.digits)
    UID += random.choice("123456789")
    return UID

