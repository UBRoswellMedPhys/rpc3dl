# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 02:26:06 2023

@author: johna
"""

import h5py
import copy
import cv2
import numpy as np
import scipy.ndimage.interpolation as scipy_mods

def rotate(original, degree_range=15, seed=None, degrees=None):
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
    
    if seed is None:
        seed = np.random.randint(1000)
    
    array = copy.deepcopy(original)

    for channel in range(3):
        subarray = array[...,channel]
        voidval = np.amin(subarray)
        np.random.seed(seed)
        intensity = np.random.random()
        if degrees is None:
            degrees = intensity*degree_range*2 - degree_range
        subarray = scipy_mods.rotate(
            subarray,
            angle=degrees,
            axes=(1,2),
            reshape=False,
            mode='constant',
            cval=voidval
            )
        array[...,channel] = subarray
        
    
    return array

def shift(original, max_shift=0.2, seed=None, pixelshift=None):
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
    
    max_y_pix = max_shift * original.shape[1]
    max_x_pix = max_shift * original.shape[2]
    
    if seed is None:
        seed = np.random.randint(1000)
    
    
    array = copy.deepcopy(original)

    for channel in range(3):
        subarray = array[...,channel]
        voidval = np.amin(subarray)
        np.random.seed(seed)
        y_intensity = np.random.random()
        x_intensity = np.random.random()

        if pixelshift is None:
            yshift = round(y_intensity*max_y_pix*2 - max_y_pix)
            xshift = round(x_intensity*max_x_pix*2 - max_x_pix)
            shiftspec = (0, yshift, xshift)
        else:
            shiftspec = (0, pixelshift[0], pixelshift[1])
            
        subarray = scipy_mods.shift(
            subarray,
            shift=shiftspec,
            mode='constant',
            cval=voidval
            )
        array[...,channel] = subarray
    

    return array
    
def zoom(original, max_zoom_factor=0.2, seed=None, zoom_factor=None):
    
    if seed is None:
        seed = np.random.randint(1000)
    
   
    array = copy.deepcopy(original)
    
    
    for channel in range(3):
        subarray = array[...,channel]
        voidval = np.amin(subarray)
        np.random.seed(seed)
        intensity = np.random.random()
        
        if zoom_factor is None:
            zoom_factor = 1 + intensity*max_zoom_factor*2 - max_zoom_factor
        
        original_shape = subarray.shape

        subarray = scipy_mods.zoom(
            subarray,
            zoom=[zoom_factor,zoom_factor,zoom_factor],
            mode='constant',
            cval=voidval
            )

        if zoom_factor > 1.0:
            subarray = bounding_box(subarray, original_shape)
        if zoom_factor < 1.0:
            diffs = np.array(original_shape) - np.array(subarray.shape)
            odd_val_offset = diffs % 2
            diffs = diffs // 2
            pad_spec = [
                (diffs[0],diffs[0] + odd_val_offset[0]),
                (diffs[1],diffs[1] + odd_val_offset[1]),
                (diffs[2],diffs[2] + odd_val_offset[2])
                ]
        
            subarray = np.pad(
                subarray,
                pad_width=pad_spec,
                mode='constant',
                constant_values=voidval
                )
        array[...,channel] = subarray
            
    return array

def bounding_box(array, shape, center=None):
    
    # allow for 2D bounding box - implied that every slice is desired
    if len(shape) == 2:
        shape = [array.shape[0]] + list(shape)
        
    # if center is none, bounding box centered around center of array
    if center is None:
        center = [
            array.shape[0] // 2,
            array.shape[1] // 2,
            array.shape[2] // 2
            ]
    else:
        center = [round(pos) for pos in center]
    start = [
        center[0] - (shape[0] // 2),
        center[1] - (shape[1] // 2),
        center[2] - (shape[2] // 2)
        ]
    
    boxed = array[
        start[0]:start[0]+shape[0],
        start[1]:start[1]+shape[1],
        start[2]:start[2]+shape[2]
        ]
    return boxed
