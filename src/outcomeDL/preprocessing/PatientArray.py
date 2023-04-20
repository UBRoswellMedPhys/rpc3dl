# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 01:02:13 2023

@author: johna
"""

import copy

import cv2
import numpy as np

import scipy.ndimage.interpolation as scipy_mods

import _preprocess_util as util

class PatientArray:
    """
    Base class to support array operations for CT, mask, dose patient arrays.
    Allows definition of common operations for all three. No __init__ method is
    defined for the base class as each modality has unique requirements on
    __init__ - this means that the base class cannot be meaningfully used on
    its own. Only the child classes should be used in practice.
    """
    
    def __init__(self, ref_file):
        self.studyUID = ref_file.StudyInstanceUID
        self.FoR = ref_file.FrameOfReferenceUID
        self.pixel_size = ref_file.PixelSpacing
    
    @property
    def position(self):
        return self._position

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
        self._position = list(np.array(self.position) - position_adjust)
        if hasattr(self, "slice_ref") and hasattr(other,"slice_ref"):
            self.slice_ref = other.slice_ref
        
    def bounding_box(self, shape, center=None):
        
        # allow for 2D bounding box - implied that every slice is desired
        if len(shape) == 2:
            shape = [self.shape[0]] + list(shape)
            
        # if center is none, bounding box centered around center of array
        if center is None:
            center = [
                self.array.shape[0] // 2,
                self.array.shape[1] // 2,
                self.array.shape[2] // 2
                ]
        else:
            center = [round(pos) for pos in center]
        start = [
            center[0] - (shape[0] // 2),
            center[1] - (shape[1] // 2),
            center[2] - (shape[2] // 2)
            ]
        
        boxed = self.array[
            start[0]:start[0]+shape[0],
            start[1]:start[1]+shape[1],
            start[2]:start[2]+shape[2]
            ]
        return boxed
    
    def rotate(self, degree_range=15, seed=None, degrees=None):
        """
        Function to rotate 3D array about the Z axis
        
        Parameters
        ----
        degree_range : int
            Max degrees +/- of rotation. Ignored if degrees is specified.
        seed : int
            Randomizer seed, used for reproducibility when transforming
            multiple related arrays. If None, seed will not be reset.
        degrees : int
            Used if you want to manually define the rotation.
        """
        if not hasattr(self,"original"):
            self.original = copy.deepcopy(self.array)
        
        if seed is not None:
            np.random.seed(seed)
        intensity = np.random.random()
        if degrees is None:
            degrees = intensity*degree_range*2 - degree_range
        self.array = scipy_mods.rotate(
            self.array,
            angle=degrees,
            axes=(1,2),
            reshape=False,
            mode='constant',
            cval=self.voidval
            )
    
    def shift(self, max_shift=0.2, seed=None, pixelshift=None):
        """
        Function which shifts array along two dimensions (does not shift in
        Z axis)

        Parameters
        ----------
        max_shift : float, optional
            Value between 0.0 and 1.0 to represent the amount of the original
            array that can max shift - so a value of 1.0 would allow the array
            to be shifted completely off (not recommended). The default is 0.2.
        seed : int, optional
            Sets np.random seed to generate shift amounts. If None, then seed
            is not specified. The default is None.
        pixelshift : tuple, optional
            Exact pixel numbers to shift by. This should correspond to the
            [Y, X] axes. This overrides the random value generator.
            The default is None.
        """
        if not hasattr(self,"original"):
            self.original = copy.deepcopy(self.array)
        
        max_y_pix = max_shift * self.rows
        max_x_pix = max_shift * self.columns
        
        if seed is not None:
            np.random.seed(seed)
        y_intensity = np.random.random()
        x_intensity = np.random.random()
        
        if pixelshift is None:
            yshift = round(y_intensity*max_y_pix*2 - max_y_pix)
            xshift = round(x_intensity*max_x_pix*2 - max_x_pix)
            shiftspec = (0, yshift, xshift)
        else:
            shiftspec = (0, pixelshift[0], pixelshift[1])
            
        self.array = scipy_mods.shift(
            self.array,
            shift=shiftspec,
            mode='constant',
            cval=self.voidval
            )
        
    def zoom(self, max_zoom_factor=0.2, seed=None, zoom_factor=None):
        if not hasattr(self,"original"):
            self.original = copy.deepcopy(self.array)
        
        if seed is not None:
            np.random.seed(seed)
        intensity = np.random.random()
        
        if zoom_factor is None:
            zoom_factor = 1 + intensity*max_zoom_factor*2 - max_zoom_factor
        
        original_shape = self.array.shape
        
        self.array = scipy_mods.zoom(
            self.array,
            zoom=[zoom_factor,zoom_factor,zoom_factor],
            mode='constant',
            cval=self.voidval
            )
        
        if zoom_factor > 1.0:
            self.array = self.bounding_box(original_shape)
        if zoom_factor < 1.0:
            diffs = np.array(original_shape) - np.array(self.array.shape)
            diffs = np.round(diffs / 2).astype(int)
            pad_spec = [
                (diffs[0],diffs[0]),
                (diffs[1],diffs[1]),
                (diffs[2],diffs[2])
                ]
            self.array = np.pad(
                self.array,
                pad_width=pad_spec,
                mode='constant',
                constant_values=self.voidval
                )
            
    def reset_augments(self):
        self.array = copy.deepcopy(self.original)
        del self.original
        
        
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
        self.voidval = -1000
        #self.studyUID = filelist[0].StudyInstanceUID
        #self.FoR = filelist[0].FrameOfReferenceUID
        #self.pixel_size = filelist[0].PixelSpacing
        super().__init__(filelist[0])
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
        self._position = sortedlist[0][1].ImagePositionPatient
        self.slice_ref = zs
        if len(np.unique(np.diff(zs))) != 1:
            self.even_spacing = False
        else:
            self.even_spacing = True
        # fill in array
        for i, (z, file) in enumerate(sortedlist):
            self.array[i,:,:] = util.getscaledimg(file)
        
    def window_level(self, window, level, normalize=False):
        upper = level + round(window/2)
        lower = level - round(window/2)
        
        self.array[self.array > upper] = upper
        self.array[self.array < lower] = lower
        
        if normalize:
            # min-max standardization, puts all values between 0.0 and 1.0
            self.array -= lower
            self.array = self.array / window

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
        
        self.voidval = 0
        self.dose_units = ref_file.DoseUnits
        super().__init__(ref_file)
        self._position = ref_file.ImagePositionPatient
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
        
        # we don't use super().__init__() for PatientMask due to the unique
        # structure of ss files
        self.voidval = 0
        self.studyUID = ssfile.StudyInstanceUID
        self.FoR = reference.FoR
        if self.studyUID != reference.studyUID:
            print("Warning: Reference file and ss file StudyUID mismatch")
        self.pixel_size = reference.pixel_size
        self._position = reference.position
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
            
    @property
    def array(self):
        return self._array
    
    @array.setter
    def array(self, value):
        """
        SciPy spline interpolation functions that are used for data
        augmentation tasks do not preserve the [0, 1] value set for the array,
        instead converting into weird floats. This enforces that any update to
        the array value remain a [0, 1] pseudo-boolean array.
        """
        self._array = value
        self._array = np.clip(self._array,0,1)
        self._array = np.abs(np.round(self._array))
        self._array = self.array.astype(np.int16)
            
    def join(self, other):
        assert isinstance(other, PatientMask)
        assert self.array.shape == other.array.shape
        self.array = self.array + other.array
        self.array[self.array > 0] = 1
        
    @property
    def com(self):
        livecoords = np.argwhere(self.array)
        com = np.sum(livecoords,axis=0) / len(livecoords)
        return com
        
if __name__ == "__main__":
    import os
    import pydicom
    testdir = r"D:\H_N\017_055"
    filepaths = [os.path.join(testdir,file) for file in os.listdir(testdir) if file.startswith("CT")]
    files = [pydicom.dcmread(file) for file in filepaths]
    dosefile = pydicom.dcmread(r"D:\H_N\017_055\RD.017_055.56-70.dcm")
    ssfile = pydicom.dcmread(r"D:\H_N\017_055\RS.017_055.CT_1.dcm")
    
    test = PatientCT(files)
    test.rescale(2.5)
    dose = PatientDose(dosefile)
    mask_l = PatientMask(test,ssfile,"Parotid (Left)")
    mask_r = PatientMask(test,ssfile,"Parotid (Right)")
    mask_l.join(mask_r)
    masks = mask_l
    dose.align_with(test)