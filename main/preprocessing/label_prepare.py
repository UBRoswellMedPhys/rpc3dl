# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 12:48:33 2022

@author: johna
"""

import pandas as pd
import os
import numpy as np

from datetime import datetime

# ==== FIRST CODE BLOCK ===
"""
This initial block of code was used to prepare round 1 labels. For these, I had
the time since treatment finished, so I used that to enforce early xerostomia
categorization (< 3 months since end of treatment)

On the broader dataset I do not at this time have detailed timing data for
all of these surveys, so I need to open up how I generate labels a little.

We will see how it works. Saving this code block for record-keeping purposes.
"""
# survey = pd.read_csv(
#     r"D:\H_N\HeadAndNeckQOLSurvey-General_anon_with_time_since.csv"
#     )

# patients = os.listdir(r"D:\extracteddata")

# summary_df = pd.DataFrame(index=patients)

# summary_df = summary_df.join(survey.set_index('ANON_ID'))

# xero_results = summary_df[['time_since_last_TX','dry_mouth','sticky_saliva']]

# threemonth_df = pd.DataFrame(index=patients,columns=xero_results.columns)
# for patient in patients:
#     subset = xero_results[(xero_results.index==patient)]

#     times = subset['time_since_last_TX'].values
#     threemonth = np.argmin(np.abs(times-30))
#     threemonth_df.loc[patient] = subset.iloc[threemonth]
    
# threemonth_df = threemonth_df.dropna()


# patientlist = []
# labellist = []
# for i,row in threemonth_df.iterrows():
#     if row['time_since_last_TX'] > 90:
#         continue
#     patientlist.append(row.name)
#     if all((row['dry_mouth'] > 2, row['sticky_saliva'] > 2)):
#         labellist.append(1)
#     elif any((row['dry_mouth'] == 4, row['sticky_saliva'] == 4)):
#         labellist.append(1)
#     else:
#         labellist.append(0)
# labels_df = pd.DataFrame(data={'patient':patientlist,'xero':labellist})

# print("Pos:",labels_df['xero'].sum())
# print("Neg:",len(labels_df) - labels_df['xero'].sum())

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
        if timediff > 90:
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