# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 23:08:42 2022

@author: johna
"""

import numpy as np
import math
import os
import pandas as pd
import scipy

from pydicom.dataset import FileDataset

class ShapeError(BaseException):
    pass

def getscaledimg(file):
    image = file.pixel_array.astype(np.int16)
    slope = file.RescaleSlope
    intercept = file.RescaleIntercept
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return image

def sort_coords(coords):
    origin = coords.mean(axis=0)
    refvec = [0,1]
    def clockwiseangle_and_dist(point):
        nonlocal origin
        nonlocal refvec
        # Vector between point and the origin: v = p - o
        vector = [point[0]-origin[0], point[1]-origin[1]]
        # Length of vector: ||v||
        lenvector = math.hypot(vector[0], vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -math.pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to subtract them 
        # from 2*pi (360 degrees)
        if angle < 0:
            return 2*math.pi+angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance should come first.
        return angle, lenvector
    sorted_coords = sorted(coords, key=clockwiseangle_and_dist)
    return np.array(sorted_coords)

def get_contour(ss,ROI):
    # retrieves ContourSequence for the requested ROI
    # accepts either name or number
    try:
        ROI_num = int(ROI)
    except ValueError:
        for info in ss.StructureSetROISequence:
            if info.ROIName == ROI:
                ROI_num = info.ROINumber
                
    for contourseq in ss.ROIContourSequence:
        if contourseq.ReferencedROINumber == ROI_num:
            if hasattr(contourseq, "ContourSequence"):
                return contourseq.ContourSequence
            else:
                return None
        
def find_parotid_info(ss,side):
    for roi in ss.StructureSetROISequence:
        name = roi.ROIName.lower()
        if 'parotid' in name:
            strippedname = name.split('parotid')
            if any(('stem' in elem for elem in strippedname)):
                continue
            if any((side in elem for elem in strippedname)):
                return roi.ROIName, roi.ROINumber
    return None, None

def find_PTV_info(ss):
    store = ("", None)
    for roi in ss.StructureSetROISequence:
        name = roi.ROIName.lower()
        if 'ptv' in name:
            if '70' in name:
                return roi.ROIName, roi.ROINumber
            else:
                store = (roi.ROIName, roi.ROINumber)
    if '56' in store[0]:
        return store
    return (None, None)

def merge_doses(*args):
    shape = None
    mergedarray = None
    for dose in args:
        if not isinstance(dose,FileDataset):
            raise TypeError("Merge doses function can only operate on" +
                            " pydicom FileDataset objects")
        if dose.Modality != 'RTDOSE':
            raise ValueError("Merge doses function can only operate on" +
                             "dose file objects.")
        if dose.DoseSummationType != 'BEAM':
            with dose.DoseSummationType as e:
                raise ValueError("Merge doses only intended to be applied" +
                                 "to beam dose files, file is {}".format(e))
        if not shape:
            shape = dose.pixel_array.shape
            ipp = dose.ImagePositionPatient
            iop = dose.ImageOrientationPatient
        else:
            if not all((
                    dose.pixel_array.shape == shape,
                    dose.ImagePositionPatient == ipp,
                    dose.ImageOrientationPatient == iop
                    )):
                raise ValueError("Mismatched arrays - cannot merge dose files")
        if mergedarray is None:
            mergedarray = dose.pixel_array * dose.DoseGridScaling
        else:
            mergedarray += dose.pixel_array * dose.DoseGridScaling
    return mergedarray

def same_shape(dicom_list):
    shapes = []
    for file in dicom_list:
        shapes.append(file.pixel_array.shape)
    if len(set(shapes)) > 1:
        shapematch = False
    else:
        shapematch = True
    return shapematch

    
def attr_shared(dcms,attr):
    # assert that all files in a list share same value for attr
    for i in range(1,len(dcms)):
        result = (getattr(dcms[0],attr) == getattr(dcms[i],attr))
        if result is False:
            break
    return result

def backfill_labels(ds,patientID,labelsfolder,condition_descriptor):
    """
    Bespoke function (not for general use) to backfill labels for multiple
    label sets. Takes h5py File object as dataset and assumes that each
    label file follows the naming convention:
        {timing}_xero_label.csv
    """
    groupname = 'labels'
    i = 1
    while groupname in ds.keys():
        i += 1
        groupname = 'labels_{}'.format(i)
    lblgrp = ds.create_group(groupname)
    lblgrp.attrs['desc'] = condition_descriptor
    for file in os.listdir(labelsfolder):
        if file.endswith("xero_label.csv"):
            timing = file.split("_")[0]
            lbl_df = pd.read_csv(os.path.join(labelsfolder,file),index_col=0)
            lbl_df.index = lbl_df.index.astype(str)
            if str(patientID) in lbl_df.index:
                labelvalue = lbl_df.loc[str(patientID),'label']
            else:
                labelvalue = 99
            lblgrp.attrs[timing] = labelvalue
            
def unpack_mask(f,key):
    dense = np.zeros_like(f['ct'][...])
    slices = f[key]['slices'][...]
    slice_nums = np.unique(slices).astype(int)
    for sl in slice_nums:
        rows = f[key]['rows'][np.where(slices==sl)]
        cols = f[key]['cols'][np.where(slices==sl)]
        sparse = scipy.sparse.coo_matrix(
            (np.ones_like(cols),(rows,cols)),
            shape=f['ct'][...].shape[1:],
            dtype=int
            )
        dense[sl,...] = sparse.todense()
    return dense

def pack_mask(densemask):
    # represent mask in sparse format
    row = np.array([])
    col = np.array([])
    slic = np.array([])
    for i in range(densemask.shape[0]):
        sp = scipy.sparse.coo_matrix(densemask[i,...])
        sl = np.full(sp.data.shape,fill_value=i,dtype=np.int32)
        slic = np.concatenate([slic,sl])
        col = np.concatenate([col,sp.col])
        row = np.concatenate([row,sp.row])    
    return slic, row, col