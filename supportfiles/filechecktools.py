# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 22:06:59 2022

@author: johna
"""

import numpy as np
import cv2
import math

def getscaledimg(file):
    image = file.pixel_array.astype(np.int16)
    slope = file.RescaleSlope
    intercept = file.RescaleIntercept
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return image

def get_slices(image, dose, mask=None):
    dosearray = dose.pixel_array * float(dose.DoseGridScaling)
    imagearray = getscaledimg(image)
    
    image_pos = image.ImagePositionPatient[-1]
    z_list = np.array(dose.GridFrameOffsetVector) + dose.ImagePositionPatient[-1]
    dose_slice_idx = np.squeeze(np.argwhere(z_list == image_pos))
    dose_slice = dosearray[dose_slice_idx,:,:]
    
    Xcorner = image.ImagePositionPatient[0]
    Ycorner = image.ImagePositionPatient[1]
    
    imgXcoords = np.arange(Xcorner, Xcorner + (image.Rows*image.PixelSpacing[0]),image.PixelSpacing[0])
    imgYcoords = np.arange(Ycorner, Ycorner + (image.Columns*image.PixelSpacing[1]),image.PixelSpacing[1])
    doseminX = dose.ImagePositionPatient[0]
    doseminY = dose.ImagePositionPatient[1]
    dosemaxX = dose.ImagePositionPatient[0] + dose.PixelSpacing[0]*dose.pixel_array.shape[2]
    dosemaxY = dose.ImagePositionPatient[1] + dose.PixelSpacing[1]*dose.pixel_array.shape[1]
    imgXkeep = np.squeeze(np.argwhere((doseminX < imgXcoords)&(imgXcoords < dosemaxX)))
    imgYkeep = np.squeeze(np.argwhere((doseminY < imgYcoords)&(imgYcoords < dosemaxY)))
    
    trimmedimage = imagearray[imgYkeep,:]
    trimmedimage = trimmedimage[:,imgXkeep]

    #newimage = cv2.resize(trimmedimage,(dose_slice.shape[1],dose_slice.shape[0]),interpolation=cv2.INTER_AREA)
    newdose = cv2.resize(dose_slice, (trimmedimage.shape[1],trimmedimage.shape[0]),interpolation=cv2.INTER_AREA)
    
    if mask is None:
        return trimmedimage, newdose
    else:
        trimmedmask = mask[imgYkeep,:]
        trimmedmask = trimmedmask[:,imgXkeep]
        return trimmedimage, newdose, trimmedmask

def list_contours(ss):
    # lists all ROI Names 
    all_contours = {}
    for each in ss.StructureSetROISequence:
        all_contours[each.ROIName] = each.ROINumber
    return all_contours

def get_contour(ss,ROI):
    # retrieves ContourSequence for the requested ROI, default is BODY. accepts either name or number
    try:
        ROI_num = int(ROI)
    except ValueError:
        for info in ss.StructureSetROISequence:
            if info.ROIName == ROI:
                ROI_num = info.ROINumber
                
    for contourseq in ss.ROIContourSequence:
        if contourseq.ReferencedROINumber == ROI_num:
            return contourseq.ContourSequence
        
def pull_single_slice(seq, image):
    # gets contour coords for slice corresponding to image (by z position)
    z = image.ImagePositionPatient[-1]
    for plane in seq:
        if plane.ContourData[2] == z:
            coords = np.reshape(plane.ContourData, (int(len(plane.ContourData)/3),3))
            coords = np.array([coords[:,0],coords[:,1]])
            coords = np.transpose(coords)
            return coords
    return None

def coords_to_mask(coords, image):
    Xcorner = image.ImagePositionPatient[0]
    Ycorner = image.ImagePositionPatient[1]
    imgXcoords = np.arange(Xcorner, Xcorner + (image.Rows*image.PixelSpacing[0]),image.PixelSpacing[0])
    imgYcoords = np.arange(Ycorner, Ycorner + (image.Columns*image.PixelSpacing[1]),image.PixelSpacing[1])
    
    mask = np.zeros_like(image.pixel_array)
    for point in coords:
        X_pos = np.argmin(np.abs(imgXcoords - point[0]))
        Y_pos = np.argmin(np.abs(imgYcoords - point[1]))
        mask[Y_pos,X_pos] = 1
        
    points = np.array(np.where(mask))
    points = np.array([points[1,:], points[0,:]]).T
    filledmask = cv2.fillPoly(mask,pts=[sort_coords(points)],color=1)
    # note - only works with congruous single volume ROIs. should be fine for now.
    return filledmask

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

def score_slices(img,dose,bodymask,witharrays=False):
    binary_dose = dose > 0.5
    binary_im = img > -900
    binary_mask = (bodymask == 1)
    # calculate dose not in body contour
    airdose = binary_dose ^ binary_mask
    airdose[~binary_dose] = False
    airdose_px = np.sum(airdose) # number of pixels of meaningful dose not contained in body contour - we expect zero
    # now check contour against air
    bodyair = binary_mask ^ binary_im
    bodyair[~binary_mask] = False
    percent_bodyair = round(100*np.sum(bodyair)/np.sum(binary_mask),3)
    if witharrays == False:
        return airdose_px, percent_bodyair
    else:
        return airdose_px, percent_bodyair, (airdose,bodyair)
    
def full_eval(image,dose,ss,contour='BODY',witharrays=False):
    contours = get_contour(ss,contour)
    coords = pull_single_slice(contours,image)
    mask = coords_to_mask(coords, image)
    testim, testdose,testmask = get_slices(image,dose,mask)
    return score_slices(testim,testdose,testmask,witharrays)
        