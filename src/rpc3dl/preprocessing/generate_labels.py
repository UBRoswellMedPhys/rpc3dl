# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 00:01:22 2023

@author: johna

Script to generate labels from QOL survey file and patient info file.
"""

import os
import pandas as pd
import argparse

from rpc3dl.preprocessing.nondicomclasses import (
    Condition,
    Survey,
    PatientInfo
    )


def main():
    parser = argparse.ArgumentParser(
        description="Script to process CSVs of survey responses and pt chars."
        )
    parser.add_argument(
        "survey_path", type=str, help="Path to QOL survey file"
        )
    parser.add_argument(
        "pt_chars_path", type=str, help="Path to patient data DB file"
        )
    parser.add_argument(
        "dest_dir", type=str, help="Path to directory to save output files to"
        )
    parser.add_argument(
        '-lm',
        '--label_mode',
        type=str,
        default='majority',
        help="Label mode: 'majority', 'all', or 'any'"
        )
    parser.add_argument(
        '-d',
        '--days_cutoff',
        type=int,
        default=90,
        help="Number of days post treatment to distinguish early vs late xero"
        )
    parser.add_argument(
        '-tc',
        '--time_column',
        type=str,
        default="eortc_qlqc30_35_timestamp,RT Completion Date",
        help="Column name or comma-separated list of names that define time columns"
        )
    parser.add_argument(
        '-id',
        '--id_column',
        type=str,
        default="MRN,ANON_ID",
        help="Column name or comma-separated list of names that define patient ID columns"
        )
    parser.add_argument(
        '-c',
        '--condition',
        type=str,
        default="dry_mouth >= 3,sticky_saliva >= 3",
        help="String descriptor of positive label condition (ex. 'dry_mouth >= 3')"
        )
    
    args = parser.parse_args()
    
    # =====================
    # Set script variables from command line arguments
    # =====================
    
    QOL_PATH = args.survey_path
    PT_CHARS_PATH = args.pt_chars_path
    DEST_DIR = args.dest_dir
    CONDITIONS = [x.strip() for x in args.condition.split(",")]
    LABEL_MODE = args.label_mode
    TIME_CUTOFF = args.days_cutoff
    TIME_COLS = [x.strip() for x in args.time_column.split(",")]
    ID_COLS = [x.strip() for x in args.id_column.split(",")]
    
    
    SAVE = True # lets me manually toggle for debug purposes
    
    # =====================
    # Prepare Condition
    # =====================
    full_condition = None
    for i, cond in enumerate(CONDITIONS):
        x = Condition(*cond.split(" "))
        if full_condition is None:
            full_condition = x
        else:
            full_condition = full_condition & x
    
    # =====================
    # Retrieve CSV files
    # =====================
    qol = pd.read_csv(QOL_PATH)
    qol = Survey(
        qol,
        time_col = TIME_COLS,
        id_col = ID_COLS
        )
    
    db = pd.read_csv(PT_CHARS_PATH)
    db = PatientInfo(
        db,
        id_col = ID_COLS,
        time_col = TIME_COLS
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
            towrite.to_csv(
                os.path.join(DEST_DIR,f'{time}_xero_label.csv')
                )
            
        # Also prepare the patient characteristics and save those
        db.scrub_data()
        db.to_csv(os.path.join(DEST_DIR,"pt_char_scrubbed.csv"))
        
        
if __name__ == "__main__":
    main()