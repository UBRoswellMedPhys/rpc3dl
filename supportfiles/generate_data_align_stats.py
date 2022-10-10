# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 22:16:24 2022

@author: johna
"""

import pydicom
import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt

import filechecktools as util

parser = argparse.ArgumentParser()
parser.add_argument('folder',nargs=1,type=str)
args = parser.parse_args()

resultspath = "dataval2.csv"

try:
    results = pd.read_csv(resultspath)
    image_idx = results['imageID'].max()
except FileNotFoundError:
    results = pd.DataFrame(columns=['Patient Folder','imgfile','dosefile','ssfile','airdose_px','bodyair_%','imageID','notes'])
    image_idx = 0

parentdir = args.folder[0]

for subdir in os.listdir(parentdir):
    if subdir in results['Patient Folder']:
        continue
    temppath = os.path.join(parentdir,subdir)
    if not os.path.isdir(temppath):
        continue
    files = os.listdir(temppath)
    if not any(([file.endswith('.dcm') for file in files])):
        continue
    dosefiles = [os.path.join(temppath,file) for file in files if file.startswith('RD')]
    ssfiles = [os.path.join(temppath,file) for file in files if file.startswith('RS')]
    imgfiles = [os.path.join(temppath,file) for file in files if file.startswith('CT')]
    
    if len(dosefiles) == 0:
        results = results.append({'Patient Folder':subdir,'notes':'NO DOSE'},ignore_index=True)
    if len(ssfiles) == 0:
        results = results.append({'Patient Folder':subdir,'notes':'NO SS'},ignore_index=True)
    
    for sspath in ssfiles:
        ss = pydicom.read_file(sspath)
        contours = util.list_contours(ss)
        contours = {k.lower(): v for k,v in contours.items()}
        if 'body' not in contours.keys():
            results = results.append({'Patient Folder': subdir,'ssfile':sspath,'notes':'NO BODY CONTOUR'},ignore_index=True)
        else:
            num = contours['body']
        for dosepath in dosefiles:
            dose = pydicom.read_file(dosepath)
            for imagepath in imgfiles:
                image = pydicom.read_file(imagepath)
                try:
                    airdose_px, bodyairperc, arrays = util.full_eval(image,dose,ss,contour=num,witharrays=True)
                except:
                    results = results.append({"Patient Folder": subdir,
                                              "imgfile": imagepath,
                                              "dosefile": dosepath,
                                              "ssfile": sspath,
                                              "notes": "INCOMPATIBILITY - UNKNOWN"},ignore_index=True)
                    continue
                resultdict = {'Patient Folder':subdir,
                              'imgfile':imagepath,
                              'dosefile':dosepath,
                              'ssfile':sspath,
                              'airdose_px':airdose_px,
                              'bodyair_%':bodyairperc}
                if airdose_px > 2000 or bodyairperc > 1.0:
                    fig,ax = plt.subplots(2,1)
                    fig.suptitle(subdir+" - "+imagepath)
                    ax[0].imshow(arrays[0])
                    ax[1].imshow(arrays[1])
                    plt.savefig('datavalfigs/{}.png'.format(image_idx))
                    plt.close()
                    resultdict['imageID'] = image_idx
                    resultdict['notes'] = 'needs review'
                    image_idx += 1
                results = results.append(resultdict,ignore_index=True)
    results.to_csv(resultspath,index=False)
    print("Done with {}".format(subdir))