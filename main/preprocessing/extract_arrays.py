# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 22:34:07 2022

@author: johna
"""

import traceback

import pydicom
import cv2
import numpy as np

import matplotlib.pyplot as plt

import _preprocess_util as util

class ShapeError(BaseException):
    pass

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
        raise ShapeError("Mismatching image sizes")
    
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

def prepare_mask_array(imgarray,slicemap,corner,ss,oar,pixel_size=1):
    mask = np.zeros_like(imgarray)
    
    # below clause is only universal if orientation is orthogonal
    # but we're requiring that, so should be fine
    realX = np.arange(corner[0], corner[0] + imgarray.shape[1], pixel_size)
    realY = np.arange(corner[1], corner[1] + imgarray.shape[2], pixel_size)
    
    seq = util.get_contour(ss,oar) # accepts ROI num
    if seq is None:
        return mask
    coords_dict = {}
    for plane in seq:
        z = plane.ContourData[2]
        coords = np.reshape(
            plane.ContourData,
            (int(len(plane.ContourData)/3),3)
            )
        coords = np.array([coords[:,0],coords[:,1]])
        coords = np.transpose(coords)
        coords_dict[z] = coords
    
    for i in range(len(imgarray)):
        z = slicemap[i]
        if z in coords_dict.keys():
            mask_slice = np.zeros((imgarray.shape[1],imgarray.shape[2]))
            for point in coords_dict[z]:
                # finds closest pixel to assign coordinate into
                X_pos = np.argmin(np.abs(realX - point[0]))
                Y_pos = np.argmin(np.abs(realY - point[1]))
                mask_slice[Y_pos,X_pos] = 1
            points = np.array(np.where(mask_slice))
            points = np.array([points[1,:],points[0,:]]).T
            mask_slice = cv2.fillPoly(mask_slice,
                                      pts=[util.sort_coords(points)],
                                      color=1)
            mask[i,:,:] = mask_slice
    if np.sum(mask) == 0:
        print("WTF")
    return mask



def prepare_dose_array(dosefile_or_list,pixel_size=1):
    if isinstance(dosefile_or_list,list):
        raw_array = util.merge_doses(*dosefile_or_list)
        dosefile = dosefile_or_list[0]
    else:
        dosefile = dosefile_or_list
        raw_array = dosefile.pixel_array
    assert dosefile.ImageOrientationPatient == [1,0,0,0,1,0]
    z_list = np.array(dosefile.GridFrameOffsetVector)
    true_z = z_list + dosefile.ImagePositionPatient[-1]
    
    new_shape = util.new_slice_shape(dosefile,pixel_size)
    change = True
    if all((new_shape[0]==raw_array.shape[2],
            new_shape[1]==raw_array.shape[1])):
        change = False
    
    # dose array shape is [slices, rows, columns] - this matters because
    # cv2 resize takes a 2D shape input descriptor as (columns, rows)
    rescaled_slices_list = []
    for i in range(len(raw_array)):
        single_slice = raw_array[i,:,:]
        if change is True:
            rescaled_slice = cv2.resize(
                single_slice.astype(float),
                (util.new_slice_shape(dosefile,pixel_size)),
                 interpolation=cv2.INTER_LINEAR
                 )
        else:
            rescaled_slice = single_slice
        rescaled_slices_list.append(rescaled_slice)
    newdose = np.array(rescaled_slices_list)
    corner = np.round(dosefile.ImagePositionPatient[0:2],1).tolist()
    true_z = true_z.tolist()
    return newdose, true_z, corner
    
    
    

if __name__ == '__main__':
    parent_dir = r"F:\DICOMdata\RoswellData"
    dest_dir = r"D:\extracteddata"
    
    pixel_size = 1
    
    import os
    import json
    for subdir in os.listdir(parent_dir):
        if not subdir.startswith("018_040"):
            continue
        testfolder = os.path.join(parent_dir,subdir)
        if not os.path.isdir(testfolder):
            continue
        print("Processing {}".format(subdir))
        imgfiles = []
        dosefile = []
        ssfile = None
        for file in os.listdir(testfolder):
            if file.startswith("CT"):
                imgfiles.append(pydicom.read_file(os.path.join(testfolder,file)))
            elif file.startswith("RD"):
                dosefile.append(pydicom.read_file(os.path.join(testfolder,file)))
            elif file.startswith("RS"):
                if "BODY" in file:
                    continue
                ssfile = pydicom.read_file(os.path.join(testfolder,file))
        
        if any((len(dosefile) == 0, ssfile is None)):
            print("Skipping {} due to missing file".format(testfolder))
            continue
        try:
            imgarr, im_slicemap, im_corner = prepare_image_array(imgfiles,pixel_size)
        except AssertionError:
            print("Issue with data validation for {}".format(testfolder))
            continue

        except ShapeError:
            print("Skipping {} due to different shape images".format(testfolder))
            continue
        
        if len(dosefile) == 1:
            dosefile = dosefile[0]
        try:
            dosearr, d_slicemap, d_corner = prepare_dose_array(dosefile,pixel_size)
        except ValueError as exc:
            print(traceback.format_exc())
            print(exc)
            continue
        parotid_r_num = util.find_parotid_num(ssfile,'r')
        parotid_l_num = util.find_parotid_num(ssfile,'l')
        if any((parotid_r_num is None, parotid_l_num is None)):
            print("Skipping {} due to missing contour".format(testfolder))
            continue
        par_r_mask = prepare_mask_array(
            imgarr,im_slicemap,im_corner,ssfile,parotid_r_num,pixel_size
            )
        par_l_mask = prepare_mask_array(
            imgarr,im_slicemap,im_corner,ssfile,parotid_l_num,pixel_size
            )
        savefolder = os.path.join(dest_dir,subdir)
        if not os.path.exists(savefolder):
            os.mkdir(savefolder)
        

        im_header = {'z_list':im_slicemap,
                      'corner_coord':im_corner,
                      'pixel_size_mm':pixel_size}
        np.save(os.path.join(savefolder,"CT.npy"),imgarr)
        with open(os.path.join(savefolder,"CT_metadata.json"),"w+") as f:
            json.dump(im_header,f)
            f.close()
        
        # dose
        d_header = {'z_list':d_slicemap,
                    'corner_coord':d_corner,
                    'pixel_size_mm':pixel_size}
        np.save(os.path.join(savefolder,"dose.npy"),dosearr)
        with open(os.path.join(savefolder,"dose_metadata.json"),"w+") as f:
            json.dump(d_header,f)
            f.close()
        
        np.save(os.path.join(savefolder,"parotid_r_mask.npy"),par_r_mask)
        np.save(os.path.join(savefolder,"parotid_l_mask.npy"),par_l_mask)