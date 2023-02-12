# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 12:48:33 2022

@author: johna
"""

import pandas as pd
import os
import numpy as np

outcome_target = 'dry_mouth'
time_bin_target = 'late' # 'early' or 'late'
fit_to_prepared_data = True
prepared_data_path = '/home/johnasbach/Research/arrays'


survey = pd.read_csv(
    "/home/johnasbach/Research/HNQOLsurvey.csv"
    )
survey = survey.dropna(subset=['MRN'])
survey['MRN'] = survey['MRN'].astype(int)

# correct non-int values in time since treatment (NaN or #VALUE!)
for i, row in survey.iterrows():
    try:
        survey.at[i,'DAYS_SINCE_TX_COMP'] = int(survey.at[i,'DAYS_SINCE_TX_COMP'])
    except:
        survey.at[i,'DAYS_SINCE_TX_COMP'] = 999

if fit_to_prepared_data:
    patients = os.listdir(prepared_data_path) # assumes subdirs are MRNs
else:
    patients = list(survey['MRN'].unique())

summary_df = pd.DataFrame(index=patients)
summary_df['label'] = 999

for patient in patients:
    subset = survey[survey['MRN']==int(patient)]
    subset = subset[subset['DAYS_SINCE_TX_COMP']!=999]
    subset = subset.dropna(subset=[outcome_target])
    if len(subset) == 0:
        continue
    for i, row in subset.iterrows():
        timediff = row['DAYS_SINCE_TX_COMP']
        if timediff > 90:
            time_bin = 'late'
        elif timediff <= 90:
            time_bin = 'early'
        if time_bin != time_bin_target:
            continue
        
        if row[outcome_target] > 2:
            summary_df.at[patient,'label'] = 1
            # overwrite default value then break row loop and go to next pt
            # if ANY survey result points positive, we accept positive as the
            # label value
            break
        else:
            summary_df.at[patient,'label'] = 0

summary_df = summary_df[summary_df['label']!=999]

if len(summary_df) != len(patients):
    difference = len(patients) - len(summary_df)
    print("{} of {} patients without valid label data".format(
        difference,len(patients)
        ))
summary_df.index = summary_df.index.rename("MRN")
summary_df.to_csv(
    "/home/johnasbach/Research/{}_{}_label.csv".format(
        time_bin_target, outcome_target
    )
)