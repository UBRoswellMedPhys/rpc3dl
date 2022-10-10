# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 12:48:33 2022

@author: johna
"""

import pandas as pd
import os
import numpy as np

survey = pd.read_csv(
    r"D:\H_N\HeadAndNeckQOLSurvey-General_anon_with_time_since.csv"
    )

patients = os.listdir(r"D:\extracteddata")

summary_df = pd.DataFrame(index=patients)

summary_df = summary_df.join(survey.set_index('ANON_ID'))

xero_results = summary_df[['time_since_last_TX','dry_mouth','sticky_saliva']]

threemonth_df = pd.DataFrame(index=patients,columns=xero_results.columns)
for patient in patients:
    subset = xero_results[(xero_results.index==patient)]

    times = subset['time_since_last_TX'].values
    threemonth = np.argmin(np.abs(times-30))
    threemonth_df.loc[patient] = subset.iloc[threemonth]
    
threemonth_df = threemonth_df.dropna()


patientlist = []
labellist = []
for i,row in threemonth_df.iterrows():
    if row['time_since_last_TX'] > 90:
        continue
    patientlist.append(row.name)
    if all((row['dry_mouth'] > 2, row['sticky_saliva'] > 2)):
        labellist.append(1)
    elif any((row['dry_mouth'] == 4, row['sticky_saliva'] == 4)):
        labellist.append(1)
    else:
        labellist.append(0)
labels_df = pd.DataFrame(data={'patient':patientlist,'xero':labellist})

print("Pos:",labels_df['xero'].sum())
print("Neg:",len(labels_df) - labels_df['xero'].sum())