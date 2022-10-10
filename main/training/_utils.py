# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 22:06:36 2022

@author: johna
"""

import os
import json
import numpy as np
import math

def gen_inputs(labels,epochs,batch_size,shuffle=True,single=False,**kwargs):
    patientlist_init = list(labels.keys())
    patientlist = []
    if shuffle is True:
        for i in range(epochs):
            np.random.shuffle(patientlist_init)
            patientlist += patientlist_init
    elif shuffle is False:
        patientlist = patientlist_init * epochs
    
    X = []
    Y = []
    for patientID in patientlist:        
        folder = "D://extracteddata//" + patientID
        dose = np.load(os.path.join(folder,"dose.npy"))
        img = np.load(os.path.join(folder,"CT.npy"))
        with open(os.path.join(folder,"dose_metadata.json")) as f:
            dose_info = json.load(f)
            f.close()
        with open(os.path.join(folder,"CT_metadata.json")) as f:
            im_info = json.load(f)
            f.close()
        par_r = np.load(os.path.join(folder,"parotid_r_mask.npy"))
        par_l = np.load(os.path.join(folder,"parotid_l_mask.npy"))
        try:
            expanded_dose = dose_expand(img,dose,im_info,dose_info)
        except ValueError:
            print("Review patient {}".format(patientID))
            continue
        Y = np.array(labels[patientID])[np.newaxis]
        
        if single is False:
            X_left, X_right = prep_inputs(img,
                                          expanded_dose,
                                          par_l,
                                          par_r,
                                          single=single,
                                          **kwargs)
            
            yield (X_left[np.newaxis,...], X_right[np.newaxis,...]), Y
        elif single is True:
            X = prep_inputs(img,
                            expanded_dose,
                            par_l,
                            par_r,
                            single=single,
                            **kwargs)
            yield X[np.newaxis,...], Y

def prep_inputs(img,dose,par_l,par_r,
                box_shape=(40,90,60),
                wl=True,normalize=True,
                dose_norm=False,
                withmask=False,
                masked=True,
                ipsi_contra=False,
                single=False):
    # note that dose must be expanded to fill image array space
    margin0 = round(box_shape[0] / 2)
    margin1 = round(box_shape[1] / 2)
    margin2 = round(box_shape[2] / 2)
    merged_mask = par_l + par_r
    
    if masked:
        img = img * merged_mask
        dose = dose * merged_mask
    
    if wl:
        img = window_level(img,normalize=normalize)
    
    if dose_norm:
        dose = dose_scaling(dose)
    if single is False:
        com_l = np.round(mask_com(par_l)).astype(int)
        com_r = np.round(mask_com(par_r)).astype(int)
        
        img_l = img[com_l[0]-margin0:com_l[0]+margin0,
                    com_l[1]-margin1:com_l[1]+margin1,
                    com_l[2]-margin2:com_l[2]+margin2]
        img_r = img[com_r[0]-margin0:com_r[0]+margin0,
                    com_r[1]-margin1:com_r[1]+margin1,
                    com_r[2]-margin2:com_r[2]+margin2]
        dose_l = dose[com_l[0]-margin0:com_l[0]+margin0,
                      com_l[1]-margin1:com_l[1]+margin1,
                      com_l[2]-margin2:com_l[2]+margin2]
        dose_r = dose[com_r[0]-margin0:com_r[0]+margin0,
                      com_r[1]-margin1:com_r[1]+margin1,
                      com_r[2]-margin2:com_r[2]+margin2]
        mask_l = par_l[com_l[0]-margin0:com_l[0]+margin0,
                       com_l[1]-margin1:com_l[1]+margin1,
                       com_l[2]-margin2:com_l[2]+margin2]
        mask_r = par_r[com_r[0]-margin0:com_r[0]+margin0,
                       com_r[1]-margin1:com_r[1]+margin1,
                       com_r[2]-margin2:com_r[2]+margin2]
        
        if withmask:
            left = (img_l,dose_l,mask_l)
            right = (img_r,dose_r,mask_r)
        else:
            left = (img_l,dose_l)
            right = (img_r,dose_r)
        left = np.stack(left,axis=-1)
        right = np.stack(right,axis=-1)
        
        if ipsi_contra:
            if np.sum(right[...,1]) > np.sum(left[...,1]):
                ipsi = np.flip(right,axis=2)
                contra = np.flip(left,axis=2)
                return ipsi, contra
            
        return left, right
    
    elif single is True:
        com = np.round(mask_com(merged_mask)).astype(int)
        
        img = img[com[0]-margin0:com[0]+margin0,
                  com[1]-margin1:com[1]+margin1,
                  com[2]-margin2:com[2]+margin2]
        dose = dose[com[0]-margin0:com[0]+margin0,
                    com[1]-margin1:com[1]+margin1,
                    com[2]-margin2:com[2]+margin2]
        mask = merged_mask[com[0]-margin0:com[0]+margin0,
                           com[1]-margin1:com[1]+margin1,
                           com[2]-margin2:com[2]+margin2]
        if withmask:
            to_return = (img, dose, mask)
        else:
            to_return = (img, dose)
        to_return = np.stack(to_return,axis=-1)
        return to_return

def dose_expand(img,dose,im_info,dose_info):
    assert dose_info['pixel_size_mm'] == im_info['pixel_size_mm']
    dose_arr = np.zeros_like(img,dtype=np.float64)
    #find number of pixels difference
    x_diff = dose_info['corner_coord'][0] - im_info['corner_coord'][0]
    x_diff = math.floor(x_diff)
    if x_diff == 0 and dose.shape[2] > dose_arr.shape[2]:
        dose = dose[:,:,0:dose_arr.shape[2]]
        
    y_diff = dose_info['corner_coord'][1] - im_info['corner_coord'][1]
    y_diff = math.floor(y_diff)
    if y_diff == 0 and dose.shape[1] > dose_arr.shape[1]:
        dose = dose[:,:,0:dose_arr.shape[1]]
    
    z_min = np.squeeze(np.argwhere(np.array(im_info['z_list'],dtype=np.float32) == float(dose_info['z_list'][0])))
    minadjust = 0
    while z_min.size == 0:
        minadjust += 1
        z_min = np.squeeze(np.argwhere(np.array(im_info['z_list'],dtype=np.float32) == float(dose_info['z_list'][0+minadjust])))
    z_max = np.squeeze(np.argwhere(np.array(im_info['z_list'],dtype=np.float32) == float(dose_info['z_list'][-1])))
    maxadjust = 0
    while z_max.size == 0:
        maxadjust += 1
        z_max = np.squeeze(np.argwhere(np.array(im_info['z_list'],dtype=np.float32) == float(dose_info['z_list'][-1-maxadjust])))
    
    dose_arr[z_min:z_max+1,
             y_diff:min(y_diff+dose.shape[1],dose_arr.shape[1]),
             x_diff:min(x_diff+dose.shape[2],dose_arr.shape[2])] = dose[minadjust:dose.shape[0] - maxadjust,:,:]
    return dose_arr

def mask_com(mask):
    livecoords = np.argwhere(mask)
    if livecoords.size == 0:
        print("Empty mask provided")
        return None
    com = np.sum(livecoords,axis=0) / len(livecoords)
    return com

def window_level(array,window=400,level=50,normalize=False):
    # default is soft tissue filter with no normalization
    upper = level + round(window/2)
    lower = level - round(window/2)
    
    newarray = array.copy()
    newarray[newarray > upper] = upper
    newarray[newarray < lower] = lower
    
    if normalize:
        # min-max standardization, puts all values between 0.0 and 1.0
        newarray -= lower
        newarray = newarray / window
    return newarray

def dose_scaling(dosearray,upperlim=75):
    newdosearray = dosearray / upperlim
    return newdosearray