# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 23:08:42 2022

@author: johna
"""

import numpy as np
import math

from pydicom.dataset import FileDataset


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
        
def find_parotid_num(ss,side):
    for roi in ss.StructureSetROISequence:
        name = roi.ROIName.lower()
        if 'parotid' in name:
            strippedname = name.split('parotid')
            if any(('stem' in elem for elem in strippedname)):
                continue
            if any((side in elem for elem in strippedname)):
                return roi.ROINumber
    return None


def new_slice_shape(file,pixel_size=1):
    rows = file.Rows
    row_spacing = file.PixelSpacing[0]
    row_scaling = row_spacing / pixel_size
    cols = file.Columns
    col_spacing = file.PixelSpacing[1]
    col_scaling = col_spacing / pixel_size
    new_cols = round(cols*col_scaling)
    new_rows = round(rows*row_scaling)
    return new_cols, new_rows

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
        else:
            if dose.pixel_array.shape != shape:
                raise ValueError("Mismatched shapes - cannot merge dose files")
        if mergedarray is None:
            mergedarray = dose.pixel_array * dose.DoseGridScaling
        else:
            mergedarray += dose.pixel_array * dose.DoseGridScaling
    return mergedarray

def mask_com(mask):
    livecoords = np.argwhere(mask)
    if livecoords.size == 0:
        print("Empty mask provided")
        return None
    com = np.sum(livecoords,axis=0) / len(livecoords)
    return com