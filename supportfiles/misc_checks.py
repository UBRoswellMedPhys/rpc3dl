# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 00:31:33 2022

@author: johna
"""

import nrrd
import numpy as np
import pandas as pd
import os

import data_utils as util

# load dataframe of patientIDs needing review
# review_df = pd.read_csv('needsreview.csv')
# review_ids = review_df['patient'].unique() # list of patient IDs

# review_df['outcome'] = None
# review_df['preferred_dosefile'] = None
# review_df['preferred_ssfile'] = None

# dataval = pd.read_csv('dataval.csv')

# for i, row in review_df.iterrows():
#     if row['num_dose']==0 or row['num_ss'] == 0:
#         review_df.loc[i,'outcome'] = "RECAPTURE"
#         continue
#     patientdata = dataval[dataval['Patient Folder']==row['patient']]
#     if any((patientdata['airdose_px'].isna().sum() > len(patientdata)/2,
#             patientdata['bodyair_%'].isna().sum() > len(patientdata)/2)):
#         review_df.loc[i,'outcome'] = "DEBUG"
#     if row['num_dose'] == 1:
#         review_df.loc[i,'preferred_dosefile'] = patientdata['dosefile'].unique()[0]
#     elif row['num_dose'] == 2:
#         review_df.loc[i,'outcome'] = "RECAPTURE" #only one case of this
    
    # if row['num_ss'] == 1:
    #     review_df.loc[i,'preferred_ssfile'] = patientdata['ssfile'].unique()[0]
    #     review_df.loc[i,'outcome'] = "MANUAL REVIEW"
    # elif row['num_ss'] == 2:
    #     if any((patientdata['airdose_px'].isna().sum() > len(patientdata)/2,
    #             patientdata['bodyair %'].isna().sum() > len(patientdata)/2)):
    
    
