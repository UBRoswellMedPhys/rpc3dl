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

def gen_inputs(config, labels, ptchars, training=True):
    """
    Core data generator for neural network training. This plays a key role as
    it provides us flexibility in how we define data processing.

    Parameters
    ----------
    NOTE: any parameter marked with a * indicates that it is extracted from
    the config, but descriptions are provided here for clarity.
    
    * source_dir : str
        Path to the parent directory where the patient subdirectories reside.
    labels : dict
        Keys are ANON IDs, values are either 1 or 0. Keys are used as reference
        for loading data from files. Expectation is that each ANONID has its
        own subdirectory in the patient directory.
    * epochs : int
        Number of epochs the training will go for. This is used to duplicate
        the patient reference list this number of times so that the generator 
        can continue to produce batches for the entire training process.
    ptchars : pd.DataFrame, optional
        DataFrame of one-hot encoded patient characteristics. Default is None,
        and if None is provided then it bypasses.
    * shuffle : bool, optional
        Whether or not to shuffle data. This is used during the epoch-duplicate
        process of building the reference list. If set to True, the ANONIDs are
        shuffled for each epoch. This mirrors Keras's native shuffle parameter
        functionality for the fit method (which is incompatible with 
        generators). The default is True.
    * single : bool, optional
        Determines whether to return single-array X. If False, uses the parotid
        bounding box method, which produces two arrays for each X, one each to 
        represent left and right (or ipsilateral and contralateral). The 
        default is False.
    * class_balance : str or False, optional
        Sets the class balance mode. Acceptable values are False (no balancing),
        "oversample", and "undersample". Oversample continuously duplicates
        the ANONIDs for the small class until the population is balanced.
        The default is False.
    * batch_size : int, optional
        Required if class_balance is used to ensure that the full, balanced
        dataset is cleanly divisible by the batch size. The default is None.
        
    Yields
    ------
    X : array or tuple of arrays
        Data, either full array (if single input) or tuple of arrays
        (if multi-input)
    Y : array
        Target data. Simply is pulled from labels provided to the function,
        but is packaged with an additional axis

    """
    
    # === Get parameters from config ====
    source_dir = config.get('filepaths','source')
    if training is True:
        # add one 'fake' epoch to ensure we generate more than enough data
        epochs = config.getint('model_settings','epochs') + 1
        class_balance = config.get(
            'data_settings','class_balance',fallback='oversample'
            )
        shuffle = config.getboolean('data_settings','shuffle',fallback=True)
    elif training is False:
        # if not for training, we only want one full set
        epochs = 1
        class_balance = None
        shuffle = False

    batch_size = config.getint('model_settings','batch_size')
    single = config.getboolean('data_settings','single',fallback=False)
    
    
    
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
        folder = os.path.join(source_dir, str(patientID))

        contents = os.listdir(folder)
        if len(contents) == 1:
            # clause to allow study-level walkdown
            folder = os.path.join(folder,contents[0])
        
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
                        masks=[par_l,par_r],
                        config=config)
        if ptchars is not None:
            try:
                onehot = ptchars.loc[patientID].values.astype(np.float32)
            except KeyError:
                # TODO - figure out what to do about no chars
                onehot = np.zeros(shape=(38),dtype=np.float32)
            if not isinstance(X, tuple):
                X = tuple([X])
            X += tuple([onehot])
        yield X, Y

def prep_inputs(img,dose,masks,
                config):
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
    NOTE: parameters marked with * are pulled out of config, but descriptions
    are still provided here for clarity.
    
    img : np.array
        3D array of voxel data
    dose : np.array
        3D array of dose data
    masks : list of np.array
        List of 3D array of masks, binary array. Must all be same shape. Checks
        against "single" parameter from config - if single is True, then all
        mask arrays will be merged to one unified mask for evaluation. If
        single is False, then each array will be leveraged independently
    * wl : bool, optional
        Whether to apply window/level processing on the image
        voxel data. The default is True.
    * normalize : bool, optional
        Whether to normalize the image data. If True, requires 
        wl == True to function. The default is True.
    * dose_norm : bool, optional
        Whether or not to normalize dose values to 0.0-1.0.
        The default is False.
    * withmask : bool, optional
        Whether to include mask array as a third channel of the returned 
        inputs. The default is True.
    * masked : bool, optional
        Whether to zero out all voxels not in the parotid (via pixelwise mult).
        The default is False.
    * ipsi_contra : bool, optional
        Whether to set the high-dose parotid to be "left" - functionally making
        the left side always ipsilateral and the right side always 
        contralateral. This is done by summing the dose voxels in the parotid
        regions. The default is False.
    * single : bool, optional
        Whether to return single-X (True --> whole head) or dual-X 
        (False --> boxed around parotids). The default is False.
    * augment : bool, optional
        Whether to perform augmentation (rotations, zooms). 
        The default is False.

    Returns
    -------
    to_return : tuple
        Tuple of arrays for volumetric data

    """
    
    # === Get parameters from config ===
    single = config.getboolean('data_settings','single',fallback=False)
    wl = config.getboolean('data_settings','wl',fallback=True)
    wl_window = config.getint('data_settings','wl_window',fallback=400)
    wl_level = config.getint('data_settings','wl_level',fallback=50)
    normalize = config.getboolean('data_settings','normalize',fallback=True)
    dose_norm = config.getboolean('data_settings','dose_norm',fallback=False)
    withmask = config.getboolean('data_settings','withmask',fallback=True)
    masked = config.getboolean('data_settings','masked',fallback=False)
    ipsi_contra = config.getboolean(
        'data_settings','ipsi_contra',fallback=False
        )
    augment = config.getboolean('data_settings','augment',fallback=False) 
    
    # === Handle mask inputs ===
    if isinstance(masks,list):
        masks = np.array(masks)
        # axis 0 is the mask sequence, rest of the axes are volume dimensions
    merged_mask = np.sum(masks,axis=0)
    
    # === Validate inputs, notify user of incompatibilities ===
    if all((wl is False, normalize is True)):
        # normalize will be ignored
        print("Voxel normalization can only occur with window/level filtering.")
        print("Normalization will not be performed.")
    if all((single is True, ipsi_contra is True)):
        print("Ipsi-contra standardization with single-array return not yet " +
              "supported. Ipsi-contra standardization will be ignored.")
    
    # prep box shape and necessary margins to crop around centers of mass
    # TODO - make box shape configurable
    if single is True:
        box_shape = (40,256,256)
    elif single is False:
        box_shape = (50,128,128)
    margin0 = round(box_shape[0] / 2)
    margin1 = round(box_shape[1] / 2)
    margin2 = round(box_shape[2] / 2)
    
    if masked:
        # zeros out all voxels not in a parotid
        img = img * merged_mask
        dose = dose * merged_mask
    
    if wl:
        # TODO - include configurable window/level
        img = window_level(img,
                           window=wl_window,
                           level=wl_level,
                           normalize=normalize)
    
    if dose_norm:
        dose = dose_scaling(dose)
        
    """
    Output is generated based off list of masks. Iterates through masks and
    generates appropriate 4D channeled arrays for each. If single is True, then
    masks are unified and a list of length 1 is created. Only one array is then
    returned.
    """
    if single is True:
        # if single-array return is desired, we replace the list of masks
        # with a single-item list of the merged mask created previously
        masks = [merged_mask]
        
    output = []
    for mask in masks:
        # trims arrays to boxes around center of mass of mask being considered
        com = mask_com(mask)
        if com is not None:
            com = np.round(mask_com(mask)).astype(int)
        shaped_img = img[com[0]-margin0:com[0]+margin0,
                         com[1]-margin1:com[1]+margin1,
                         com[2]-margin2:com[2]+margin2]
        shaped_dose = dose[com[0]-margin0:com[0]+margin0,
                           com[1]-margin1:com[1]+margin1,
                           com[2]-margin2:com[2]+margin2]
        shaped_mask = mask[com[0]-margin0:com[0]+margin0,
                           com[1]-margin1:com[1]+margin1,
                           com[2]-margin2:com[2]+margin2]
        if withmask:
            arraystuple = (shaped_img,shaped_dose,shaped_mask)
        else:
            arraystuple = (shaped_img,shaped_dose)
        
        init_array = np.stack(arraystuple, axis=-1)
        # now have an array of axes (Z,X,Y,channels)
        
        if augment is True:
            if np.random.random() > 0.5:
                seed = np.random.random()
                init_array = rotation(init_array,seed)
            if np.random.random() > 0.5:
                seed = np.random.random()
                # zoom is way faster on 3D arrays than 4D, so we'll
                # deconstruct back to 3D components then re-stack
                templist = [
                    init_array[...,i] for i in range(init_array.shape[-1])
                    ]
                for i in range(len(templist)):
                    templist[i] = zoom_aug(templist[i],seed=seed)
                init_array = np.stack(templist,axis=-1)
        
        output.append(init_array)
                
    if ipsi_contra:
        if len(output) == 2:
            """
            If 2 arrays, checks dose channel (last axis, position 1) sum to
            determine which array is higher dose. If necessary, flips the
            order and the orientation of the arrays to put higher dose first.
            """
            if np.sum(output[1][...,1]) > np.sum(output[0][...,1]):
                ipsi = np.flip(output[1],axis=2)
                contra = np.flip(output[0],axis=2)
                output = [ipsi, contra]
        elif len(output) == 1:
            """
            If 1 array, splits it down the middle and checks cumulative dose
            on either side to determine ipsi vs contra. Positions higher-dose
            side on the left.
            """
            halfway = output[0].shape[2] // 2
            leftdose = output[0][:,:,:halfway,1]
            rightdose = output[0][:,:,halfway:,1]
            if np.sum(rightdose) > np.sum(leftdose):
                output[0] = np.flip(output[0],axis=2)
        else:
            raise Exception(
                "Number of output arrays {}".format(len(output)) +
                " incompatible with ipsi_contra standardization."
                )
    output = [array.astype(np.float32) for array in output]
    
    if len(output) == 1:
        to_return = output[0]
    else:
        to_return = tuple(output)
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
    
    # assert compatible pixel sizing
    assert dose_info['pixel_size_mm'] == im_info['pixel_size_mm']
    
    # instantiate an image-array size-match empty array
    dose_arr = np.zeros_like(img,dtype=np.float64)
    
    #find number of pixels difference
    x_diff = dose_info['corner_coord'][0] - im_info['corner_coord'][0]
    x_diff = math.floor(x_diff)
    y_diff = dose_info['corner_coord'][1] - im_info['corner_coord'][1]
    y_diff = math.floor(y_diff)
    
    # sometimes rounding of the float creates issues, if shapes match, we
    # enforce diff of 0 pixels
    if dose.shape[2] == img.shape[2] and x_diff == 1:
        x_diff = 0 
    if dose.shape[1] == img.shape[1] and y_diff == 1:
        y_diff = 0 
        
    # if front edge of either axis is aligned and dose array is larger than
    # destination array, we can simply trim off from 0 index
    if x_diff == 0 and dose.shape[2] > dose_arr.shape[2]:
        dose = dose[:,:,0:dose_arr.shape[2]]
    if y_diff == 0 and dose.shape[1] > dose_arr.shape[1]:
        dose = dose[:,0:dose_arr.shape[1],:]
    
    # dose z list in order from min to max
    mindex = 0
    maxdex = 0
    dose_zmin = float(dose_info['z_list'][mindex])
    dose_zmax = float(dose_info['z_list'][maxdex-1])
    im_z_list = np.array(im_info['z_list'],dtype=np.float32)
    
    # if dose z-axis is fully contained by image array (it should be) then
    # no adjustment is needed. however, if dose array extends beyond image
    # array, we need to walk in the min-max boundaries to "drop" the overhang
    while dose_zmin not in im_z_list:
        mindex += 1
        dose_zmin = float(dose_info['z_list'][mindex])
    while dose_zmax not in im_z_list:
        maxdex -= 1
        dose_zmax = float(dose_info['z_list'][maxdex-1])
        
    # finalize the array indices of the min and max
    z_min = np.squeeze(np.argwhere(im_z_list == dose_zmin))
    z_max = np.squeeze(np.argwhere(im_z_list == dose_zmax))
    
    # configure the slicing for filling destination array from source array
    z_dest = range(z_min,z_max+1)
    z_source = range(mindex,dose.shape[0] + maxdex)
    z_source, z_dest = match_slice_size(z_source,z_dest)
    
    if y_diff >= 0:
        y_dest = range(
            y_diff,min(y_diff+dose.shape[1],dose_arr.shape[1])
            )
        y_source = range(dose.shape[1])
    elif y_diff < 0:
        y_dest = range(dose_arr.shape[1])
        y_source = range(-y_diff,-y_diff+img.shape[1])
    y_source, y_dest = match_slice_size(y_source,y_dest)
    
    if x_diff >= 0:
        x_dest = range(
            x_diff,min(x_diff+dose.shape[2],dose_arr.shape[2])
            )
        x_source = range(dose.shape[2])
    elif x_diff < 0:
        x_dest = range(dose_arr.shape[2])
        x_source = range(-x_diff,-x_diff+img.shape[2])
    x_source, x_dest = match_slice_size(x_source,x_dest)
        
    dose_arr[z_dest,y_dest,x_dest] = dose[z_source,y_source,x_source]

    return dose_arr

def match_slice_size(source,dest):
    """
    Function to ensure array slicing for an axis is of the same length.
    This is required for broadcasting. This assumes start-aligned voxels,
    so if there is a mismatch then the longer range will be trimmed down.
    
    Parameters
    ----------
    source : range
        Range of source index to be converted to slice
    dest : range
        Range of destination index to be converted to slice
        
    Returns
    -------
    source_slice : slice
        Fitted slice, ready for pass into array slicing
    dest_slice : slice
        Fitted slice, ready for pass into array slicing
    """
    length = min(len(source),len(dest)) # find shortest range
    source_slice = slice(source.start,source.start+length)
    dest_slice = slice(dest.start,dest.start+length)
    return source_slice, dest_slice

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


if __name__ == '__main__':
    for i in [1,2,3,4]:
        debugfolder = r"D:\arrays_to_review\{}".format(i)
        #debugfolder = r"D:\extracteddata\018_002"
        img = np.load(os.path.join(debugfolder,'CT.npy'))
        dose = np.load(os.path.join(debugfolder,'dose.npy'))
        with open(os.path.join(debugfolder,'ct_metadata.json')) as f:
            im_info = json.load(f)
            f.close()
        with open(os.path.join(debugfolder,'dose_metadata.json')) as f:
            dose_info = json.load(f)
            f.close()
        x = dose_expand(img, dose, im_info, dose_info)
        print(i)
        print(x.shape)
        