# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 12:48:33 2022

@author: johna
"""

import pandas as pd
import os
import numpy as np



survey = pd.read_csv(
    r"D:\H_N\HNQOLSurvey_NewAnon_w_days_since_TXCOMP.csv"
    )

# correct non-int values in time since treatment (NaN or #VALUE!)
for i, row in survey.iterrows():
    try:
        survey.at[i,'DAYS_SINCE_TX_COMP'] = int(survey.at[i,'DAYS_SINCE_TX_COMP'])
    except:
        survey.at[i,'DAYS_SINCE_TX_COMP'] = 999

patients = list(survey['ANONID'].unique())

summary_df = pd.DataFrame(index=patients)
summary_df['OLDID'] = None
summary_df['label'] = 999

for patient in patients:
    subset = survey[survey['ANONID']==patient]
    subset = subset[subset['DAYS_SINCE_TX_COMP']!=999]
    subset = subset.dropna(subset=[
        'dry_mouth','sticky_saliva'
        ])
    if len(subset) == 0:
        continue
    for i, row in subset.iterrows():
        timediff = row['DAYS_SINCE_TX_COMP']
        if timediff < 150:
            # reject 'late' surveys
            continue
        
        if not row.isna()['OLDANON_ID']:
            summary_df.at[patient,'OLDID'] = row['OLDANON_ID']
        if all((row['dry_mouth'] > 2, row['sticky_saliva'] > 2)) or \
            any((row['dry_mouth'] == 4, row['sticky_saliva'] == 4)):
            summary_df.at[patient,'label'] = 1
            # overwrite default value then break row loop and go to next pt
            # if ANY survey result points positive, we accept positive as the
            # label value
            break
        else:
            summary_df.at[patient,'label'] = 0

summary_df = summary_df[summary_df['label']!=999]