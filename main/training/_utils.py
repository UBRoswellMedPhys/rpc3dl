# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 22:06:36 2022

@author: johna
"""

import os
import json
import numpy as np
import math

from scipy.ndimage.interpolation import rotate, shift, zoom

def batcher(generator,batch_size,num_inputs=3):
    """
    Wrapper function which allows single-instance generator output to be
    packaged into batches so that it can be used as feed for NN training
    
    Parameters
    ----------
    generator : Generator
        Data generator object that yields single-instance (X,Y) on each next()
    batch_size : int
        Number of X, Y pairs to package into a batch
    num_inputs : int
        Number of input arrays in X
        
    Yields
    ------
    X,Y : np.array
        Batched X,Y pairs of length = batch_size
    """
    # TODO - will need to update this to support patient characteristic inclusion
    
    # instantiate temp variables to batch building
    X = {i:[] for i in range(num_inputs)}
    Y = []
    while True:
        if len(Y) == batch_size:
            # resets our X,Y if we just sent a full batch out
            X = {i:[] for i in range(num_inputs)}
            Y = []
        nextX, nextY = next(generator)
        # TODO -- Unsure if this code works with single array, revisit later
        for i,array in enumerate(nextX):
            X[i].append(array)
        Y.append(nextY)
        if len(Y) == batch_size:
            # batch is ready, package in a NN-friendly format
            X_final = [np.stack(X[i],axis=0) for i in range(num_inputs)]
            X_final = tuple(X_final)
            Y = np.array(Y)
            yield X_final, Y

def gen_inputs(source_dir,labels,epochs,ptchars=None,shuffle=True,single=False,
               class_balance=False,batch_size=None,**kwargs):
    """
    Core data generator for neural network training. This plays a key role as
    it provides us flexibility in how we define data processing.

    Parameters
    ----------
    source_dir : str
        Path to the parent directory where the patient subdirectories reside.
    labels : dict
        Keys are ANON IDs, values are either 1 or 0. Keys are used as reference
        for loading data from files. Expectation is that each ANONID has its
        own subdirectory in the patient directory.
    epochs : int
        Number of epochs the training will go for. This is used to duplicate
        the patient reference list this number of times so that the generator 
        can continue to produce batches for the entire training process.
    ptchars : pd.DataFrame, optional
        DataFrame of one-hot encoded patient characteristics. Default is None,
        and if None is provided then it bypasses.
    shuffle : bool, optional
        Whether or not to shuffle data. This is used during the epoch-duplicate
        process of building the reference list. If set to True, the ANONIDs are
        shuffled for each epoch. This mirrors Keras's native shuffle parameter
        functionality for the fit method (which is incompatible with 
        generators). The default is True.
    single : bool, optional
        Determines whether to return single-array X. If False, uses the parotid
        bounding box method, which produces two arrays for each X, one each to 
        represent left and right (or ipsilateral and contralateral). The 
        default is False.
    class_balance : str or False, optional
        Sets the class balance mode. Acceptable values are False (no balancing),
        "oversample", and "undersample". Oversample continuously duplicates
        the ANONIDs for the small class until the population is balanced.
        The default is False.
    batch_size : int, optional
        Required if class_balance is used to ensure that the full, balanced
        dataset is cleanly divisible by the batch size. The default is None.
    **kwargs
        Arguments to pass to the prep_inputs() function.
        
    Yields
    ------
    X : array or tuple of arrays
        Data, either full array (if single input) or tuple of arrays
        (if multi-input)
    Y : array
        Target data. Simply is pulled from labels provided to the function,
        but is packaged with an additional axis

    """
    
    # === Prepare necessary class balance values ===
    if class_balance and any((not shuffle, not batch_size)):
        raise ValueError(
            "Class balance without shuffle and batch_size not supported"
            )
    elif class_balance:
        pos = []
        neg = []
        for k,v in labels.items():
            if v == 1:
                pos.append(k)
            elif v == 0:
                neg.append(k)
        small_class_size = min((len(pos),len(neg)))
        large_class_size = max((len(pos),len(neg)))
        remainder = (len(pos) + len(neg)) % batch_size
        large_class_size -= remainder # for oversample, keep to clean batch size
    
    # === Prepare complete ID list to iterate through ===
    patientlist_init = list(labels.keys())
    patientlist = []
    for i in range(epochs):
        if shuffle is False:
            # if no shuffle, by default no class balance, so we can just
            # duplicate the ID list for each epoch
            patientlist += patientlist_init
        elif shuffle is True:
            temp = []
            if not class_balance:
                # if no class balance then it's easy, just shuffle and add
                np.random.shuffle(patientlist_init)
                patientlist += patientlist_init
            elif class_balance == "undersample":
                # TODO - ensure this returns cleanly divisible by batch size
                temp += list(np.random.choice(
                    pos,size=small_class_size,replace=False
                    ))
                temp += list(np.random.choice(
                    neg,size=small_class_size,replace=False
                    ))
                np.random.shuffle(temp)
                patientlist += temp
            elif class_balance == "oversample":
                for cl in [pos,neg]:
                    if len(cl) >= large_class_size:
                        # must constrain for batch_size divisibility
                        temp += cl[:large_class_size]
                    elif len(cl) < large_class_size:
                        # small class
                        hold = []
                        while len(hold) < large_class_size:
                            # repeatedly add all ANONIDs to a holding space
                            # until size passes large_class_size
                            hold += cl
                        temp += hold[:large_class_size]
                # temp now contains both classes and represents a full epoch
                np.random.shuffle(temp)
                patientlist += temp
            else:
                raise ValueError(
                    "Unrecognized value provided for class_balance arg"
                    )
    
    # this is the actual generator loop which loads the files and passes them
    # to the appropriate preprocessing
    for patientID in patientlist:
        # load all necessary files        
        folder = os.path.join(source_dir, patientID)
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
        
        # refit dose array to be same shape as image array - see the docstring
        # for the dose_expand function for more details
        try:
            expanded_dose = dose_expand(img,dose,im_info,dose_info)
        except ValueError as e:
            # TODO - explore cause of errors with dose_expand
            # for now we just skip patients with errors
            print("Review patient {}".format(patientID))
            print(e)
            continue
        # package the integer label value as a shape (1,1) array
        Y = np.array(labels[patientID])[np.newaxis]
        
        # call prep_inputs - note that X may be a tuple of arrays depending
        # on whether we're ina multi-input paradigm
        X = prep_inputs(img,
                        expanded_dose,
                        par_l,
                        par_r,
                        single=single,
                        **kwargs)
        if ptchars is not None:
            try:
                onehot = ptchars.loc[patientID].values.astype(np.float32)
            except KeyError:
                # TODO - figure out what to do about no chars
                onehot = np.zeros(shape=(38),dtype=np.float32)
            X += tuple([onehot])
        yield X, Y

def prep_inputs(img,dose,par_l,par_r,
                wl=True,normalize=True,
                dose_norm=False,
                withmask=True,
                masked=False,
                ipsi_contra=False,
                single=False,
                augment=False,
                **kwargs):
    """
    Function that receives the fitted arrays for image, dose, and masks and
    performs any necessary preprocessing on the data
    
    A few pre-processing steps occur prior to this function:
        1. Voxel scaling to 1mm x 1mm - this occurs in initial extraction
        from DICOM files, prior to this entire pipeline
        2. Shape fitting of dose array to match size of image array - this 
        occurs in the dose_expand() function prior to this function call

    Parameters
    ----------
    img : np.array
        3D array of voxel data
    dose : np.array
        3D array of dose data
    par_l : np.array
        3D array of Parotid L mask, binary array.
    par_r : np.array
        3D array of Parotid R mask, binary array
    wl : bool, optional
        Whether to apply window/level processing on the image
        voxel data. The default is True.
    normalize : bool, optional
        Whether to normalize the image data. If True, requires 
        wl == True to function. The default is True.
    dose_norm : bool, optional
        Whether or not to normalize dose values to 0.0-1.0.
        The default is False.
    withmask : bool, optional
        Whether to include mask array as a third channel of the returned 
        inputs. The default is True.
    masked : bool, optional
        Whether to zero out all voxels not in the parotid (via pixelwise mult).
        The default is False.
    ipsi_contra : bool, optional
        Whether to set the high-dose parotid to be "left" - functionally making
        the left side always ipsilateral and the right side always 
        contralateral. This is done by summing the dose voxels in the parotid
        regions. The default is False.
    single : bool, optional
        Whether to return single-X (True --> whole head) or dual-X 
        (False --> boxed around parotids). The default is False.
    augment : bool, optional
        Whether to perform augmentation (rotations, zooms). 
        The default is False.
    **kwargs : dict
        Not used, only for compatibility.

    Returns
    -------
    to_return : tuple
        Tuple of arrays for volumetric data

    """
    
    # === Validate inputs, notify user of incompatibilities ===
    if all((wl is False, normalize is True)):
        # normalize will be ignored
        print("Voxel normalization can only occur with window/level filtering.")
        print("Normalization will not be performed.")
    if all((single is True, ipsi_contra is True)):
        print("Ipsi-contra standardization with single-array return not yet " +
              "supported. Ipsi-contra standardization will be ignored.")
    
    # prep box shape and necessary margins to crop around centers of mass
    if single is True:
        box_shape = (40,256,256)
    elif single is False:
        box_shape = (48,108,108)
    margin0 = round(box_shape[0] / 2)
    margin1 = round(box_shape[1] / 2)
    margin2 = round(box_shape[2] / 2)
    
    merged_mask = par_l + par_r # single array with both masks
    if masked:
        # zeros out all voxels not in a parotid
        img = img * merged_mask
        dose = dose * merged_mask
    
    if wl:
        img = window_level(img,normalize=normalize)
    
    if dose_norm:
        dose = dose_scaling(dose)
        
    # TODO - split into sub-functions based on 'single' argument
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
                
        if augment == True:
            if np.random.random() > 0.5:
                seed = np.random.random()
                left = rotation(left,seed)
                right = rotation(right,seed)
                
            if np.random.random() > 0.5:
                seed = np.random.random()
                left = zoom_aug(left,seed=seed)
                right = zoom_aug(right,seed=seed)  

        
        if ipsi_contra:
            if np.sum(right[...,1]) > np.sum(left[...,1]):
                ipsi = np.flip(right,axis=2)
                contra = np.flip(left,axis=2)
                left = ipsi
                right = contra
        
        left = left.astype(np.float32)
        right = right.astype(np.float32)
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
            
        if augment == True:
            if np.random.random() > 0.5:
                seed = np.random.random()
                to_return = rotation(to_return, seed=seed)
            if np.random.random() > 0.5:
                seed = np.random.random()
                to_return = zoom_aug(to_return,seed=seed)
        
        to_return = to_return.astype(np.float32)
        return to_return

def dose_expand(img,dose,im_info,dose_info):
    """
    Function to fit the dose array to match the shape of the image array, which
    is necessary for later cropping operations. Note that we do not rescale
    voxels at all in this function, we just use the arrays and metadata to fit
    dose into image.
    
    Parameters
    ----------
    img : np.array
        3D image array
    dose : np.array
        3D dose array
    im_info : dict
        Metadata as produced by data extraction process. Required fields are:
        'pixel_size_mm', 'corner_coord', 'z_list'
    dose_info : dict
        Metadata as produced by data extraction process. Required fields are:
        'pixel_size_mm', 'corner_coord', 'z_list'
        
    Returns
    -------
    dose_arr : np.array
        Dose array that is shaped to match the shape of the image array, usable
        for cropping and other pre-processing operations.
    """
    
    assert dose_info['pixel_size_mm'] == im_info['pixel_size_mm']
    dose_arr = np.zeros_like(img,dtype=np.float64)
    #find number of pixels difference
    x_diff = dose_info['corner_coord'][0] - im_info['corner_coord'][0]
    x_diff = math.floor(x_diff)
    if dose.shape[2] == img.shape[2] and x_diff == 1:
        x_diff = 0 #little bugfix for rounding errors
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
    
    if all((x_diff >= 0, y_diff >= 0)):
        dose_arr[z_min:z_max+1,
                 y_diff:min(y_diff+dose.shape[1],dose_arr.shape[1]),
                 x_diff:min(x_diff+dose.shape[2],dose_arr.shape[2])] = dose[minadjust:dose.shape[0] - maxadjust,:,:]
    elif all((x_diff < 0, y_diff < 0)):
        dose_arr[z_min:z_max,:,:] = dose[
            minadjust:dose.shape[0] - maxadjust,
            -y_diff:-y_diff+img.shape[1],
            -x_diff:-x_diff+img.shape[2]
            ]
    elif all((x_diff < 0, y_diff >= 0)):
        dose_arr[
            z_min:z_max+1,y_diff:min(y_diff+dose.shape[1],dose_arr.shape[1]),:
                ] = dose[
                    minadjust:dose.shape[0] - maxadjust,
                    :,
                    -x_diff:-x_diff+img.shape[2]
                    ]
    elif all((x_diff >= 0, y_diff < 0)):
        dose_arr[
            z_min:z_max+1,:,x_diff:min(x_diff+dose.shape[2],dose_arr.shape[2])
            ] = dose[
                minadjust:dose.shape[0] - maxadjust,
                -y_diff:-y_diff+img.shape[1],
                :
                ]
    else:
        print(x_diff,y_diff)
        raise ValueError("Weird dimensions, X/Y are not both either smaller or larger in img/dose relationship.")
    return dose_arr

def mask_com(mask):
    """
    Function to find the center of mass of a mask volume.
    
    Parameters
    ----------
    mask : np.array
        Binary array of mask volume
        
    Returns
    -------
    com : np.array
        Three value array which indicates where the center of mass of the
        provided mask is.
    """
    livecoords = np.argwhere(mask)
    if livecoords.size == 0:
        print("Empty mask provided")
        return None
    com = np.sum(livecoords,axis=0) / len(livecoords)
    return com

def window_level(array,window=400,level=50,normalize=False):
    """
    Applies window/level filtering to image array.
    
    Parameters
    ----------
    array : np.array
        Raw image array to be filtered.
    window : int, optional
        Size of the window for the W/L filter. Default is 400.
    level : int, optional
        Level to center the window on for W/L filter. Default is 50.
    normalize : bool, optional
        Whether to normalize to 0.0-1.0 scale after applying W/L filter.
        Default is False.
        
    Returns
    -------
    newarray : np.array
        Image array with W/L applied as, if applicable, normalized.
    """
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
    """
    Applies dose normalization (min-max scaling). Since negative dose is not
    possible, the only necessary parameter is maximum possible dose.
    
    Rather than min-max scale internally, which would cause non-comparable
    absolute dose values across patients, we scale against an external upper
    limit. Default is 75.

    Parameters
    ----------
    dosearray : np.array
        Dose array to be normalized
    upperlim : int or float, optional
        Upper limit for dose values. The default is 75, since most PTVs are 
        prescribed 70 Gy, and occasionally a voxel receives over 70.

    Returns
    -------
    newdosearray : np.array
        New dose array, min-max scaled to 0.0-1.0

    """
    newdosearray = dosearray / upperlim
    return newdosearray

def rotation(array,seed=None,degree_range=15):
    """
    Function which performs rotation operation for data augmentation. Receives
    4D array of inputs and applies a rotation in the X-Y plane (maintains
    slice integrity).
    
    Parameters
    ----------
    array : np.array
        4D array to be rotated
    seed : float
        Float between 0.0 and 1.0, used to determine how much to rotate the
        image by. If none provided, generates a random float. Default is None.
    degree_range : int
        Number of degrees in each direction the max range of rotation.
        
    Returns
    -------
    array : np.array
        Rotated array.
    
    """
    if seed is None:
        seed = np.random.random()
    howmuch = seed
    howmuch *= degree_range*2
    howmuch -= degree_range
    array = rotate(
        array,angle=howmuch,axes=(1,2),reshape=False,mode='nearest'
        )
    return array

def zoom_aug(array,seed=None,zoom_range=0.1):
    """
    
    """
    if seed is None:
        seed = np.random.random()
    howmuch = seed
    howmuch *= zoom_range*2
    howmuch -= zoom_range
    howmuch += 1.0
    checkval = howmuch
    # need a clause to ensure if 4D array is processed that channels aren't zoomed
    if len(array.shape) == 4:
        howmuch = [howmuch,howmuch,howmuch,1.0]
    original_shape = array.shape
    array = zoom(array,zoom=howmuch,mode='constant',cval=0.0)
    
    if checkval > 1.0:
        array = crop_center(array,original_shape)
    elif checkval < 1.0:
        array = pad_image(array,original_shape)
    return array

def crop_center(source,endshape):      
    """
    Function to crop an array to a desired shape. End shape must be smaller
    than initial shape in all dimensions.
    
    Parameters
    ----------
    source : np.array
        Array of N dimensions
    endshape : tuple
        Desired shape. Must be same number of dimensions as source.
    
    Returns
    -------
    cropped : np.array
        Array cropped to shape of endshape
    """
    assert len(source.shape) == len(endshape), "Cannot change dimensions"
    slices = []
    for i,dim in enumerate(source.shape):
        start = dim//2-(endshape[i]//2)
        slices.append(slice(start,start+endshape[i]))
    slices = tuple(slices)
    cropped = source[slices]
    return cropped

def pad_image(source,endshape,padvalue=0.0):
    """
    Pads an array with 0.0 while retaining original data at the center for
    all dimensions. Used for when shape compatibility is required, like
    neural network training.
    
    Parameters
    ----------
    source : np.array
        Array of N dimensions, which will be padded out with zeros
    endshape : tuple
        Sequence of desired shape values. Must be the same number of dims as
        source.
    padvalue : float or int, optional
        Number to pad the array with.
        
    Returns
    -------
    new : np.array
        Padded array with original data centered.
    """
    # TODO - allow padvalue to be a sequence of values, one for each channel
    assert len(source.shape) == len(endshape), "Cannot change dimensions"
    new = np.full(endshape,padvalue)
    slices = []
    for i, dim in enumerate(source.shape):
        pad = (endshape[i] - dim) // 2
        slices.append(slice(pad,pad+dim))
    slices = tuple(slices)
    new[slices] = source
    return new