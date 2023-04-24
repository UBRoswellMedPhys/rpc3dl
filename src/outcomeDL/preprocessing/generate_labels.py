# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 00:01:22 2023

@author: johna

Script to generate labels from QOL survey file and patient info file.
"""

import os
import pandas as pd
import argparse

import _classes as helper

# =====================
# Set script variables
# =====================

QOL_PATH = "/path/to/qolsurvey.csv"
PT_CHARS_PATH = "/path/to/patientdb.csv"
CONDITIONS = [
    "dry_mouth >= 3",
    "sticky_saliva == 4"
    ]

LABEL_MODE = 'majority'
TIME_CUTOFF = 90

DEST_DIR = "/path/to/dir"
SAVE = True

# =====================
# Prepare Condition
# =====================
full_condition = None
for i, cond in enumerate(CONDITIONS):
    x = helper.Condition(*cond.split(" "))
    if full_condition is None:
        full_condition = x
    else:
        full_condition = full_condition & x

# =====================
# Retrieve CSV files
# =====================
qol = pd.read_csv(QOL_PATH)
qol = helper.Survey(
    qol,
    time_col = 'eortc_qlqc30_35_timestamp',
    id_col = 'MRN'
    )
db = pd.read_csv(PT_CHARS_PATH)
db = helper.PatientInfo(
    db,
    id_col = 'ANON_ID',
    time_col = 'RT Completion Date'
    )

unique_patients = db.data[db.id_col].unique()

# =====================
# Calculate labels for each time bracket
# =====================

wrapper_dict = {}
for time in ['acute','early','late','all']:
    label_dict = {}
    for pt in unique_patients:
        label = qol.evaluate(
            pt,
            full_condition,
            time,
            cutoff = TIME_CUTOFF,
            mode = LABEL_MODE
            )
        if label is None:
            continue
        label_dict[pt] = label
    wrapper_dict[time] = label_dict

# =====================
# Save the resulting files
# =====================

if SAVE:        
    for time,subdict in wrapper_dict.items():
        towrite = pd.DataFrame(data=subdict,index=['label']).T
        towrite.index.name = db.id_col
        towrite.to_csv(os.path.join(DEST_DIR,f'{time}_xero_label.csv'))
        
    # Also prepare the patient characteristics and save those
    db.scrub_data()
    db.to_csv(os.path.join(DEST_DIR,"pt_char_scrubbed.csv"))