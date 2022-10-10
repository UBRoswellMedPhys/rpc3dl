import numpy as np
import cv2
import math

from pydicom.dataset import FileDataset

def getscaledimg(file):
    image = file.pixel_array.astype(np.int16)
    slope = file.RescaleSlope
    intercept = file.RescaleIntercept
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return image

def get_slices(image, dose, mask=None, scaleto='img'):
    """
    Takes patient objects are returns aligned slices for each mode.
    Meta-function: calls several sub-functions. Clarity maybe isn't ideal due
    to this.

    Parameters
    ----------
    image : pydicom file
        CT image file as loaded with pydicom. Will be passed to getscaledimg()
    dose : pydicom file
        Dose file as loaded from pydicom.
    mask : np.array, optional
        Mask that matches the image file passed. The default is None.
    scaleto : str, optional
        Controls whether to scale arrays to the img scaling or to the dose 
        scaling. The default is 'img'.

    Returns
    -------
    toreturn : tuple
        Tuple of image slice, dose slice, (optional) mask slice.

    """
    # NOTE: need to figure out how to correct the variable outputs problem here
    dosearray = dose.pixel_array * float(dose.DoseGridScaling)
    imagearray = getscaledimg(image)
    
    image_pos = image.ImagePositionPatient[-1]
    z_list = np.array(dose.GridFrameOffsetVector) + dose.ImagePositionPatient[-1]
    if not np.any((z_list == image_pos)):
        print("No dose array match for z position {}".format(image_pos))
        if mask is None:
            return None, None
        else:
            return None, None, None
    dose_slice_idx = np.squeeze(np.argwhere(z_list == image_pos))
    dose_slice = dosearray[dose_slice_idx,:,:]
    
    Xcorner = image.ImagePositionPatient[0]
    Ycorner = image.ImagePositionPatient[1]
    
    imgXcoords = np.arange(
        Xcorner, Xcorner + (image.Rows*image.PixelSpacing[0]),image.PixelSpacing[0]
        )
    imgYcoords = np.arange(
        Ycorner, Ycorner + (image.Columns*image.PixelSpacing[1]),image.PixelSpacing[1]
        )
    doseminX = dose.ImagePositionPatient[0]
    doseminY = dose.ImagePositionPatient[1]
    dosemaxX = dose.ImagePositionPatient[0] + dose.PixelSpacing[0]*dose.pixel_array.shape[2]
    dosemaxY = dose.ImagePositionPatient[1] + dose.PixelSpacing[1]*dose.pixel_array.shape[1]
    imgXkeep = np.squeeze(
        np.argwhere((doseminX < imgXcoords)&(imgXcoords < dosemaxX))
        )
    imgYkeep = np.squeeze(
        np.argwhere((doseminY < imgYcoords)&(imgYcoords < dosemaxY))
        )
    
    trimmedimage = imagearray[imgYkeep,:]
    trimmedimage = trimmedimage[:,imgXkeep]
    
    if mask is not None:
        trimmedmask = mask[imgYkeep,:]
        trimmedmask = trimmedmask[:,imgXkeep]
    else:
        trimmedmask = None
    
    if scaleto == 'dose':
        trimmedimage = cv2.resize(trimmedimage,
                                  (dose_slice.shape[1], dose_slice.shape[0]),
                                  interpolation=cv2.INTER_AREA)
        if trimmedmask is not None:
            trimmedmask = cv2.resize(trimmedmask,
                                     (dose_slice.shape[1], dose_slice.shape[0]),
                                     interpolation=cv2.INTER_AREA)
    elif scaleto == 'img':
        dose_slice = cv2.resize(dose_slice, 
                                (trimmedimage.shape[1],trimmedimage.shape[0]),
                                interpolation=cv2.INTER_AREA)
    else:
        raise Exception("Invalid entry for scaleto argument")
    
    
    toreturn = tuple(
        [x for x in (trimmedimage,dose_slice,trimmedmask) if x is not None]
        )
    return toreturn

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
            if hasattr(contourseq, "ContourSequence"):
                return contourseq.ContourSequence
            else:
                return None
        
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
    
    if coords is None:
        return mask
    
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

def measure_box(fullarray):
    mask = fullarray[:,:,:,2]
    if np.sum(mask) == 0.0:
        return (0,0,0)
    indices = np.argwhere(mask)
    Zs = indices[:,0]
    Ys = indices[:,1]
    Xs = indices[:,2]
    boxshape = (np.amax(Zs) - np.amin(Zs),
                np.amax(Ys) - np.amin(Ys),
                np.amax(Xs) - np.amin(Xs))

    return boxshape

def assign_labels(patient,surveys):
    relevant_entries = surveys[surveys['ANON_ID']==patient]
    if len(relevant_entries) == 0:
        return None, None
    earlies = relevant_entries[relevant_entries['time_since_last_TX'] <= 30]
    lates = relevant_entries[relevant_entries['time_since_last_TX'] >= 360]
    earlydry = earlies['dry_mouth'].max()
    latedry = lates['dry_mouth'].max()
    earlysticky = earlies['sticky_saliva'].max()
    latesticky = lates['sticky_saliva'].max()
    if len(earlies) == 0:
        earlylabel = None
    else:
        earlylabel = int((earlydry + earlysticky)/2 > 2.5)
    if len(lates) == 0:
        latelabel = None
    else:
        latelabel = int((latedry + latesticky)/2 > 2.5)
    return earlylabel, latelabel

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

def mask_com(data,header=None):
    """
    Function to calculate center of mass of a mask array. Note that since the
    mask array is binary, the math is simplified
    
    COM equation:
        sum(xi*mi)/m, sum(yi*mi)/m, sum(zi*mi)/m
    Since all values of mi are 1, m is equal to the total sum of the array.
    Simplifies to:
        sum(xi)/sum(array), sum(yi)/sum(array), sum(zi)/sum(array)
    Where xi, yi, and zi are the indices of voxel positions

    Parameters
    ----------
    data : np.ndarray
        Data array. If array is 4D, assumed to be full array needing filtering.
        If less than 4D, assumed to be standalone mask.
    header : dict. OPTIONAL.
        Optional header from nrrd file. Used to reference channel map.

    Returns
    -------
    com : np.ndarray
        Coordinates for center of mass of mask.

    """
    if len(data.shape) < 4:
        mask = data
    else:
        maskchannel = 2 #default
        if header is not None: # override from header if present
            for i, channel in enumerate(header['channelmap']):
                if 'mask' in channel:
                    maskchannel = i
        if len(data.shape) == 4:
            mask = data[:,:,:,maskchannel] #per stand
    
    if not np.all([value in [0.,1.] for value in np.unique(mask)]):
        raise Exception("Invalid input to mask_com function, not binary")
    livecoords = np.argwhere(mask)
    if livecoords.size == 0:
        print("Empty mask provided")
        return None
    com = np.sum(livecoords,axis=0) / len(livecoords)
    return com

def split_halves(data):
    lr_len = data.shape[2]
    lefthalf = data[:,:,:int(lr_len/2),:]
    righthalf = data[:,:,int(lr_len/2):,:]
    return lefthalf, righthalf

def total_organ_dose(data):
    # calculates total parotid dose in array
    # assumes dose mapping fits [image, dose, mask] arrangement
    
    mask = data[:,:,:,2].astype(int)
    return np.sum(data[:,:,:,1][mask])

def mean_organ_dose(data):
    mask = data[:,:,:,2].astype(int)
    return np.mean(data[:,:,:,1][mask])

def set_ipsilateral_to_left(data):
    """
    Function to align ipsi/contralateral arrangement for standardization.
    If ipsilateral parotid is on the right side, horizontally flips the data.
    
    Assumption is that parotid with greater total dose is the ipsilateral.
    
    Rejects input if there isn't parotid on both sides - this assumes data
    is centered
    """
    left, right = split_halves(data)
    leftdose = total_organ_dose(left)
    rightdose = total_organ_dose(right)
    
    if rightdose > leftdose:
        return np.flip(data,axis=2)
    else:
        return data

if __name__ == '__main__':
    # print('working')
    # import pydicom
    # testimg = pydicom.read_file(r"D:\H_N\018_131\CT.018_131.Image 285.dcm")
    # testdose = pydicom.read_file(r"D:\H_N\018_131\RD.018_131.VMAT 0_56_70.dcm")
    # testss = pydicom.read_file(r"D:\H_N\018_131\RS.018_131.CT_11.1.18MAR.dcm")
    
    # full_eval(testimg, testdose, testss)
    pass