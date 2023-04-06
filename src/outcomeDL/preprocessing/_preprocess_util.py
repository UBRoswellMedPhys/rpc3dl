# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 23:08:42 2022

@author: johna
"""

import numpy as np
import math

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
    """
    Function which calculates the shape of a destination array based on
    requested pixel size. Only works with square pixels.
    
    Parameters
    ----------
    file : pydicom DICOM file object
        DICOM object - must have PixelSpacing, Rows, and Columns attributes
    pixel_size : float/int
        Value to define, in mm, the pixel size. Default is 1.
        
    Returns
    -------
    new_cols : int
        Number of columns needed in destination array
    new_rows : int
        Number of rows needed in destination array
    """
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

def correct_array_shapes(filelist):
    """
    Function which trims arrays of non-uniform shape to largest common array (fits all arrays).
    Requires that all arrays in the list have the same corner position (ImagePositionPatient attribute).
    
    Parameters
    ----------
    dosefilelist : list of DICOM files
        List containing pydicom loaded DICOM objects.
    
    Returns
    -------
    dosefilelist : list of DICOM files
        Same list, but with the files modified to be shape-compatible.
    """
    if not same_position(filelist):
        raise Exception("Files not corner-aligned, can't match them")
    # first find the smallest array - note that pixel_array.shape is structured as Z, Rows, Columns
    min_row = np.inf
    min_col = np.inf
    min_z = np.inf
    # first loop through all to find the min for each axis
    for file in filelist:
        z, rows, cols = file.pixel_array.shape
        min_row = min(rows, min_row)
        min_col = min(cols, min_col)
        min_z = min(z, min_z)
    # next restructure each array as needed
    for file in filelist:
        new_array = file.pixel_array
        if file.pixel_array.shape[0] > min_z:
            new_array = new_array[:min_z,:,:]
            file.GridFrameOffsetVector = file.GridFrameOffsetVector[:min_z]
        if file.pixel_array.shape[1] > min_row:
            new_array = new_array[:,:min_row,:]
            file.Rows = min_row
        if file.pixel_array.shape[2] > min_col:
            new_array = new_array[:,:,:min_col]
            file.Columns = min_col
        file.PixelData = new_array.tobytes()
    return filelist
        
def same_position(dicom_list):
    """
    Function to check to see if all DICOM files in a list have aligned corners.
    Returns a boolean.
    """
    c,r,z = dicom_list[0].ImagePositionPatient
    for obj in dicom_list:
        tc, tr, tz = obj.ImagePositionPatient
        if not all((tc == c, tr == r, tz == z)):
            return False
    return True

def same_shape(dicom_list):
    shapes = []
    for file in dicom_list:
        shapes.append(file.pixel_array.shape)
    if len(set(shapes)) > 1:
        shapematch = False
    else:
        shapematch = True
    return shapematch

def same_study(files):
    """
    Function which checks if files are all part of one Study. Uses the
    StudyInstanceUID attribute. Accepts either list or dict of lists,
    returns boolean.
    """
    
    if isinstance(files,list):
        filedict = {"main":files}
    elif isinstance(files,dict):
        filedict = files
    UID = []
    for key in filedict.keys():
        for file in filedict[key]:
            UID.append(file.StudyInstanceUID)
    UID = list(set(UID))
    if len(UID) > 1:
        clean = False
    elif len(UID) == 1:
        clean = True
    return clean

def same_frame_of_reference(files):
    
    if isinstance(files,dict):
        flatfiles = []
        for k in files.keys():
            flatfiles += files[k]
        files = flatfiles
    framerefs = []
    for file in files:
        if hasattr(file, "FrameOfReferenceUID"):
            framerefs.append(file.FrameOfReferenceUID)
    if len(set(framerefs)) == 1:
        sameframe = True
    else:
        sameframe = False
    return sameframe
    