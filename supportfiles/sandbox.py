# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 00:31:33 2022

@author: johna
"""

import os

import numpy as np
import nrrd
import matplotlib.pyplot as plt

import data_utils as util


nrrd_home = r"D:\xero_nrrd"

for file in os.listdir(nrrd_home):
    if file.endswith(".nrrd"):
        break
    
data, head = nrrd.read(os.path.join(nrrd_home,file))

data = data.astype(np.int8)

indices = np.argwhere(data[...,2])
Zs = indices[:,0]
Ys = indices[:,1]
Xs = indices[:,2]

Xmax = np.amax(Xs) + 5
Xmin = np.amin(Xs) - 5
Ymax = np.amax(Ys) + 5
Ymin = np.amin(Ys) - 5
Zmax = np.amax(Zs) + 5
Zmin = np.amin(Zs) - 5

small = data[Zmin:Zmax,Ymin:Ymax,Xmin:Xmax,:].copy()

fig, ax = plt.subplots(3,1,figsize=(10,7))
ax[0].imshow(data[100,...,0],cmap='gray')
ax[1].imshow(data[100,...,1])
ax[2].imshow(data[100,...,2],cmap='gray')


fig, ax = plt.subplots(figsize=(11,11))
ax.imshow(data[100,...,0],cmap='gray',zorder=1)
ax.imshow(data[100,...,1],zorder=3,alpha=0.5)
ax.imshow(data[100,...,2],cmap='gray',zorder=5,alpha=0.25)