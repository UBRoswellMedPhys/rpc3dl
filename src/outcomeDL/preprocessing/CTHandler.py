# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 01:02:13 2023

@author: johna
"""

import cv2
import numpy as np

import _preprocess_util as util

class PatientCT:
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
        

def prepare_image_array(image_files,pixel_size=1):
    """
    Function to prepare 3D array of image files.
    Scales pixel values to HU, scales image dimensions to desired pixel size.
    Assumes same-size images, since all are from the same study.
    Assumes all images are oriented orthogonally to the patient.
    Warns user if irregularities exist in slice gaps, but does not do anything
    to correct/modify slice spacing.
    
    Parameters
    ----------
    image_files : list
        List of DICOM file objects. Must all be CT.
    pixel_size : int or float
        Desired pixel size in mm. Creates square pixels. Default is 1.
    """
    temp_dict = {}
    corner = None
    shapes = []
    for file in image_files:
        # === Validate file first ===
        assert file.Modality == 'CT'
        assert file.ImageOrientationPatient == [1,0,0,0,1,0], \
            "Image not oriented orthogonally"
        z = file.SliceLocation
        if corner is None:
            corner = np.round(file.ImagePositionPatient[0:2],1).tolist()
        else:
            assert corner == np.round(
                file.ImagePositionPatient[0:2],1
                ).tolist()
        # ---- Validation complete ----
        
        img = util.getscaledimg(file)
        
        resized_img = cv2.resize(
            img, 
            (util.new_slice_shape(file,pixel_size)),
            interpolation=cv2.INTER_AREA
            )
        temp_dict[z] = resized_img
        if resized_img.shape not in shapes:
            shapes.append(resized_img.shape)
    
    if len(shapes) > 1:
        print(shapes)
        raise util.ShapeError("Mismatching image sizes")
    
    slicemap = []
    imglist = []
    for z in sorted(temp_dict):
        slicemap.append(z)
        imglist.append(temp_dict[z])
        
    imgarray = np.array(imglist) # this will error out if imgs not same size

    if len(np.unique(np.round(np.diff(slicemap),2))) > 1:
        # warn user if there's irregularity in slice gaps
        print(
            "Irregular slice thicknesses for patient {}.".format(
                file.PatientID
                )
            )
    
    return imgarray, slicemap, corner