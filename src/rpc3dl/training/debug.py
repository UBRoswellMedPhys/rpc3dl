# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 23:35:29 2022

@author: johna
"""

import numpy as np
import os
import json
import _utils as util

patients_to_check = ['018_020','018_056','018_091','018_093','018_108','018_125','018_127','ANON_016','ANON_020','ANON_021','ANON_034','ANON_035']


for patientID in patients_to_check:
    folder = "D:\\extracteddata\\" + patientID
    dose = np.load(os.path.join(folder,"dose.npy"))
    img = np.load(os.path.join(folder,"CT.npy"))
    with open(os.path.join(folder,"dose_metadata.json")) as f:
        dose_info = json.load(f)
        f.close()
    with open(os.path.join(folder,"CT_metadata.json")) as f:
        im_info = json.load(f)
        f.close()
    try:
        e_dose = util.dose_expand(img,dose,im_info,dose_info)
    except Exception as e:
        print("Error with",patientID)
        print("Dose/img shapes:",dose.shape,img.shape)
        print(e)
        continue
    if patientID == 'ANON_020':
        break