# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 01:02:13 2023

@author: johna
"""

import cv2
import numpy as np

import _preprocess_util as util

class PatientArray:            
    @property
    def height(self):
        return self.rows * self.pixel_size[0]
    
    @property
    def width(self):
        return self.columns * self.pixel_size[1]
    
    def rescale(self, new_pix_size):
        if isinstance(new_pix_size,int) or isinstance(new_pix_size, float):
            new_pix_size = (new_pix_size, new_pix_size)
        row_scaling = self.pixel_size[0] / new_pix_size[0]
        col_scaling = self.pixel_size[1] / new_pix_size[1]
        new_cols = round(self.columns * col_scaling)
        new_rows = round(self.rows * row_scaling)
        new_array = np.zeros((self.array.shape[0],new_rows,new_cols))
        for i,img in enumerate(self.array):
            rescaled = cv2.resize(
                img, 
                (new_cols, new_rows),
                interpolation=cv2.INTER_AREA
                )
            new_array[i,:,:] = rescaled
        self.array = new_array
        self.columns = new_cols
        self.rows = new_rows
        self.pixel_size = new_pix_size
        
    def locate(self, coord):
        """
        Function which accepts real-space coordinate and returns the
        indices of the appropriate voxel
        """
        x = coord[0]
        y = coord[1]
        z = coord[2]
        if any((
                z < np.amin(self.slice_ref),
                z > np.amax(self.slice_ref),
                x < self.position[0],
                x > (self.position[0] + self.columns * self.pixel_size[1]),
                y < self.position[1],
                y > (self.position[1] + self.columns * self.pixel_size[0])
                )):
            # coordinates outside of array
            return None
            
        z_idx = round(np.argmin(np.abs(np.array(self.slice_ref) - z)))
        y_idx = round((y - self.position[1]) / self.pixel_size[0])
        x_idx = round((x - self.position[0]) / self.pixel_size[1])
        return (z_idx, y_idx, x_idx)        
        
class PatientCT(PatientArray):
    def __init__(self, filelist):
        """
        PatientCT object must be instantiated on a list of CT files
        Certain metadata needs to all be the same for this to work, so we'll
        pull those off the first entry in the list, then force the rest to
        match
        
        Note that for pixel spacing attribute, the row spacing is the first
        value (Y axis) and column spacing the second value (X axis), while
        for ImagePositionPatient the values are (X, Y, Z)
        Similarly, contour triplets are (X, Y, Z).
        
        Array is stored as (Z, Y, X)
        """
        self.studyUID = filelist[0].StudyInstanceUID
        self.FoR = filelist[0].FrameOfReferenceUID
        self.pixel_size = filelist[0].PixelSpacing
        self.slice_thickness = filelist[0].SliceThickness
        self.rows = filelist[0].Rows
        self.columns = filelist[0].Columns
        zlist = []
        for file in filelist:
            if not all((
                    file.StudyInstanceUID == self.studyUID,
                    file.FrameOfReferenceUID == self.FoR,
                    file.PixelSpacing == self.pixel_size,
                    file.SliceThickness == self.slice_thickness,
                    file.Rows == self.rows,
                    file.Columns == self.columns
                    )):
                raise ValueError(
                    "Incompatible metadata in file list"
                    )
            zlist.append((float(file.ImagePositionPatient[-1]),file))
            
        sortedlist = sorted(zlist,key=lambda x: x[0])
        # create array space for image data
        self.array = np.zeros((len(filelist), self.rows,self.columns))
        zs = [tup[0] for tup in sortedlist]
        self.position = sortedlist[0][1].ImagePositionPatient
        self.slice_ref = zs
        if len(np.unique(np.diff(zs))) != 1:
            self.even_spacing = False
        else:
            self.even_spacing = True
        # fill in array
        for i, (z, file) in enumerate(sortedlist):
            self.array[i,:,:] = util.getscaledimg(file)

class PatientDose(PatientArray):
    def __init__(self, dcm):
        """
        
        Note that for pixel spacing attribute, the row spacing is the first
        value (Y axis) and column spacing the second value (X axis), while
        for ImagePositionPatient the values are (X, Y, Z)
        Similarly, contour triplets are (X, Y, Z).
        
        Array is stored as (Z, Y, X)
        """
        if not isinstance(dcm, list):
            assert dcm.DoseSummationType == "PLAN", \
                "Dose file passed is not a PLAN file, check DoseSummationType"
            ref_file = dcm
            self.array = dcm.pixel_array * dcm.DoseGridScaling
        else:
            assert all((b.DoseSummationType == "BEAM" for b in dcm)), \
                "List of DICOMs must all be BEAM files, check DoseSummationType"
            # if all are BEAM files, do a compatibility check
            mismatches = []
            for attr in [
                    "StudyInstanceUID",
                    "FrameOfReferenceUID",
                    "PixelSpacing", 
                    "GridFrameOffsetVector",
                    "Rows",
                    "Columns",
                    "DoseUnits",
                    "DoseGridScaling"
                    ]:
                if not util.attr_shared(dcm,attr):
                    mismatches.append(attr)
            if len(mismatches) > 0:
                raise Exception(
                    "Mismatched shape attributes in dose files: {}".format(
                        mismatches
                        )
                    )
            self.array = util.merge_doses(*dcm)
            # merge_doses function enforces matching of array shape
            # it also forces matching of ImagePositionPatient
            # and ImageOrientationPatient. This ensures array overlay.
            ref_file = dcm[0] # used to pull attributes
        
        self.studyUID = ref_file.StudyInstanceUID
        self.FoR = ref_file.FrameOfReferenceUID
        self.pixel_size = ref_file.PixelSpacing
        self.rows = ref_file.Rows
        self.columns = ref_file.Columns
        self.position = ref_file.ImagePositionPatient
        offset = np.array(ref_file.GridFrameOffsetVector)
        self.slice_ref = offset + self.position[-1]
        
class PatientMask(PatientArray):
    def __init__(self,reference,ssfile,roi):
        """
        Creates a mask array that is imprinted off of a reference array.
        
        Parameters
        ----------
        reference - PatientArray
        ssfile - pydicom.dataset.FileDataset
            Loaded pydicom object of the structure set
        roi - str
            Name of the ROI to create a mask of
        """
        self.studyUID = ssfile.StudyInstanceUID
        self.FoR = reference.FoR
        if self.studyUID != reference.studyUID:
            print("Warning: Reference file and ss file StudyUID mismatch")
        self.pixel_size = reference.pixel_size
        self.rows = reference.rows
        self.columns = reference.rows
        self.position = reference.position
        self.slice_ref = reference.slice_ref
        
        self.array = np.zeros_like(reference.array)
        for roi_info in ssfile.StructureSetROISequence:
            if roi_info.ROIName == roi:
                ref_num = roi_info.ROINumber
                break
        for data in ssfile.ROIContourSequence:
            if data.ReferencedROINumber == ref_num:
                contourseq = data.ContourSequence
                break
        
        for plane in contourseq:
            coords = np.reshape(
                plane.ContourData,
                (int(len(plane.ContourData)/3),3)
                )
            for point in coords:
                self.array[reference.locate(point)] = 1
        
        for i in range(self.array.shape[0]):
            mask_slice = self.array[i]
            if np.sum(mask_slice) == 0:
                continue
            points = np.array(np.where(mask_slice))
            points = np.array([points[1,:],points[0,:]]).T
            self.array[i] = cv2.fillPoly(
                mask_slice,
                pts=[util.sort_coords(points)],
                color=1
                )
            
    def join(self, other):
        assert isinstance(other, PatientMask)
        assert self.array.shape == other.array.shape
        self.array = self.array + other.array
        self.array[self.array > 0] = 1