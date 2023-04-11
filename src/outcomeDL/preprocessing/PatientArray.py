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
    def rows(self):
        return self.array.shape[1]
    
    @property
    def columns(self):
        return self.array.shape[2]
    
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
    
    def align_with(self, other):
        """
        Function which shapes data array to match shape and be aligned with
        the array of other. Common use here will be to fit a dose array onto
        an image array. Mask arrays won't need this as much since they are
        imprinted off of an array, so you can just imprint off it once your
        array is set.
        """
        assert (self.even_spacing and other.even_spacing), \
            "Cannot align arrays with uneven spacing/missing slices"
        if other.pixel_size != self.pixel_size:
            self.rescale(other.pixel_size)
        # assign slice thickness attributes if not present
        # due to previous assertion, we can use np.unique(np.diff(obj))[0]
        if not hasattr(self, "slice_thickness"):
            self.slice_thickness = np.unique(np.diff(self.slice_ref))[0]
        if not hasattr(other, "slice_thickness"):
            other.slice_thickness = np.unique(np.diff(other.slice_ref))[0]
        # find corner differences in voxel-steps
        # create voxel size list that maps to position axes (X, Y, Z)
        voxel_size = [float(self.pixel_size[1]),
                      float(self.pixel_size[0]),
                      float(self.slice_thickness)]
        front_pad = []
        for p_s, p_o, size in zip(self.position, other.position,voxel_size):
            voxel_steps = round((p_s - p_o) / size)
            front_pad.append(voxel_steps)
        
        # reminder that array shape is in reverse order: (Z, Y, X)
        # if front_pad value is positive, it means self's array needs to be
        # expanded
        back_pad = []
        back_pad.append(other.columns - (self.columns + front_pad[0])) #X
        back_pad.append(other.rows - (self.rows + front_pad[1])) #Y
        back_pad.append(len(other.array) - (len(self.array) + front_pad[2])) #Z
        
        # now we trim if any pads are negative. front pad first
        pad_value = -1000 if isinstance(self,PatientCT) else 0
        if front_pad[0] < 0:
            self.array = self.array[:,:,-front_pad[0]:]
            self.position[0] -= (front_pad[0] * self.pixel_size[1])
            front_pad[0] = 0 # set to zero as adjustment is complete
        if front_pad[1] < 0:
            self.array = self.array[:,-front_pad[1]:,:]
            self.position[1] -= (front_pad[1] * self.pixel_size[0])
            front_pad[1] = 0 # set to zero as adjustment is complete
        if front_pad[2] < 0:
            self.array = self.array[-front_pad[2]:,:,:]
            self.slice_ref = self.slice_ref[-front_pad[2]:]
            self.position[2] -= (front_pad[2] * self.slice_thickness)
            front_pad[2] = 0 # set to zero as adjustment is complete
        if back_pad[0] < 0:
            self.array = self.array[:,:,:back_pad[0]]
            back_pad[0] = 0
        if back_pad[1] < 0:
            self.array = self.array[:,:back_pad[1],:]
            back_pad[1] = 0
        if back_pad[2] < 0:
            self.array = self.array[:back_pad[2],:,:]
            self.slice_ref = self.slice_ref[:back_pad[2]]
            back_pad[2] = 0
        padarg = [(front,back) for front,back in zip(front_pad,back_pad)]
        padarg.reverse() # recall that shape is backwards order, (Z,Y,X)
        self.array = np.pad(
            self.array,
            pad_width=padarg,
            constant_values=pad_value
            )

        position_adjust = np.array(front_pad) * np.array(voxel_size)
        self.position = list(np.array(self.position) - position_adjust)
        
        
        
        
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
        refrows = filelist[0].Rows
        refcols = filelist[0].Columns
        zlist = []
        for file in filelist:
            if not all((
                    file.StudyInstanceUID == self.studyUID,
                    file.FrameOfReferenceUID == self.FoR,
                    file.PixelSpacing == self.pixel_size,
                    file.SliceThickness == self.slice_thickness,
                    file.Rows == refrows,
                    file.Columns == refcols
                    )):
                raise ValueError(
                    "Incompatible metadata in file list"
                    )
            zlist.append((float(file.ImagePositionPatient[-1]),file))
            
        sortedlist = sorted(zlist,key=lambda x: x[0])
        # create array space for image data
        self.array = np.zeros((len(filelist), refcols, refrows))
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
        self.position = ref_file.ImagePositionPatient
        offset = np.array(ref_file.GridFrameOffsetVector)
        self.slice_ref = offset + self.position[-1]
        if len(np.unique(np.diff(offset))) == 1:
            self.even_spacing = True
        else:
            self.even_spacing = False
        
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
        self.position = reference.position
        self.slice_ref = reference.slice_ref
        self.even_spacing = reference.even_spacing
        
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
        
if __name__ == "__main__":
    import os
    import pydicom
    testdir = r"D:\H_N\017_055"
    filepaths = [os.path.join(testdir,file) for file in os.listdir(testdir) if file.startswith("CT")]
    files = [pydicom.dcmread(file) for file in filepaths]
    dosefile = pydicom.dcmread(r"D:\H_N\017_055\RD.017_055.56-70.dcm")
    
    test = PatientCT(files)
    dose = PatientDose(dosefile)
    test.align_with(dose)