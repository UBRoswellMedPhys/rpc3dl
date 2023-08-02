# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 16:58:43 2023

@author: johna
"""

import pandas as pd
import re

def check_and_convert_float(column):
    try:
        # Try converting the column to float
        converted_column = pd.to_numeric(column, errors='raise')
        return converted_column
    except (TypeError, ValueError):
        initial_values = len(column.dropna())
        # If an error occurs, handle the bad values
        converted_column = pd.to_numeric(column, errors='coerce')
        # Count the non-null values
        non_null_count = converted_column.notnull().sum()
        # Check if the majority of cells can be converted
        if non_null_count >= initial_values / 2:
            # Return the converted column with bad values replaced by NaN
            return converted_column
        else:
            # Return the original column if the majority of cells cannot be converted
            return None

class Database:
    """Class for loading CSV file export from RedCap and processing it using
    a rules-based appraoch rather than prescribing hard-coded rules for each
    column.
    
    The critical aspect here is to ensure no PII is output but maximize the
    amount of data retained.
    
    Logic rules to guide:
        - Free text fields must be dropped. Not only is it virtually
        impossible to guarantee that a free text field does not contain any
        potential PII, but it also would be outrageously difficult to utilize
        the data in an encoding for DL
            > We can check for free text fields by scanning the number of
            unique values in a field. If it's too high, then we know it's not
            a multiple choice field.
            > Caveating the above, values that are numeric or dates will have
            many entries. We need to check the DATA TYPE first to know whether
            to handle the field as a number or date. If it only can be handled
            as a string, and it has too many unique values, then the field
            must be dropped.
        - Dates all need to be scrubbed. We'll need to retain them in Database
        memory while processing, because some of the information may matter.
        However, no dates can be retained (PII). We'll convert dates to
        RELATIVE times, with the reference value being the DATE OF DIAGNOSIS.
        All other dates will become "thing X time - days since diagnosis" and
        if some things happened before diagnosis then it can be represented
        as a negative, so be it.
            > Note that PATIENT AGE must be flattened to year and if that year
            is over 90, then it needs to be binned into a 90+ bin.
        - Grouped fields should be compressed. Some multiple choice fields are
        set up so that each option is its own column with "Checked/Unchecked"
        as options. We want to collect these and only retain the values of the
        "checked" responses. We can join multiple positive selections using
        a | (pipe) character.
        - As a final sweep, any field title that contains 'name', 'address',
        ('telephone','phone','cell') + ('number','#'), or 'SSN'/'social'
    
    Note an additional challenge is that SURVEY responses also have associated
    dates and it's important that we measure these against:
        - RT Start Date
        - RT Complete Date
    This means that the Database object will need to be able to process
    """
    
    def __init__(self,df,id_col="MRN",anchordate='Date of Diagnosis',
                 filters={'Diagnosis Type':'Primary','Event Name':'First Diagnosis'}):
        self.id_col = id_col
        self.anchordate = anchordate
        self.db = df
        self.date_format = '%m/%d/%Y'
        self.filters = filters
        self.fields_to_delete = [
            "TPN In",
            "TPN Out",
            "Details of third or more admissions:",
            "Other Induction Chemotherapy Frequency"
            ]
        #self.build_type_map()
        #self.clean_data()
        # all this prep can be done up front - handling grouped fields will
        # need to be done on patientID call
        
    def build_type_map(self):
        self.type_map = {}
        for col in self.db.columns:
            # first, date check
            if 'date' in col.lower():
                self.db[col] = pd.to_datetime(
                    self.db[col], format=self.date_format,errors='coerce'
                    )
                self.type_map[col] = 'date'
                continue
            # next we check for float
            check_col = check_and_convert_float(self.db[col])
            # returns None if not a valid conversion
            if check_col is not None:
                self.db[col] = check_col
                self.type_map[col] = 'float'
                continue
            # if neither date nor float works, assume string
            self.db[col] = self.db[col].astype(str)
            self.type_map[col] = 'str'
            
    def clean_data(self):
        for k,v in self.filters.items():
            self.db = self.db[self.db[k].str.strip()==v]
        self.db = self.db.set_index(self.id_col,drop=True)
        #store essential info
        for col in self.db.columns:
            if "rt duration" in col.lower():
                self.durations = self.db[col].dropna().to_dict()
            if 'rt completion date' in col.lower():
                self.completion_dates = self.db[col].dropna().to_dict()
        for col, t in self.type_map.items():
            if col == self.id_col:
                continue
            if t == 'date':
                if col == self.anchordate:
                    continue
                self.db[col] = self.db[col] - self.db[self.anchordate]
                self.db[col] = self.db[col].apply(lambda x: x.days)
            # when this is complete, the self.anchordate column is still
            # represented as an actual date, all others are the timedelta since
            # that date
            elif t == 'str':
                num_uniq = self.db[col].nunique()
                total_entries = self.db[col].count()
                if (num_uniq > 10) and ((num_uniq / total_entries) > 0.01):
                    if 'stage' in col.lower():
                        continue
                    print("Dropping {}".format(col))
                    self.db.drop(columns=col,inplace=True)
                elif col in self.fields_to_delete:
                    print("Dropping {}".format(col))
                    self.db.drop(columns=col,inplace=True)
                
            if 'age' in col.lower() and self.type_map[col] == 'float':
                self.db[col] = self.db[col].apply(
                    lambda x: x if x < 90.0 else 90.0
                    )
                
    def handle_grouped_fields(self):
        # first scan for indicator
        headers = []
        for col in self.db.columns:
            if '(choice=' in col.lower():
                header, selection = col.split("(choice=")
                header = header.strip()
                if header not in headers:
                    headers.append(header)
                selection = selection.strip(")")
                self.db[col] = self.db[col].apply(
                    lambda x: selection if x == 'Checked' else ""
                    )
        for header in headers:
            subset = [
                col for col in self.db.columns if col.strip().startswith(header)
                ]
            subset = [
                col for col in subset if col.split("(choice=")[0].strip()==header
                ]
            responses = self.db[subset].astype(str).apply("|".join,axis=1)
            responses = responses.str.strip("|") # drops leading/trailing |
            responses = responses.apply(
                lambda x: re.sub(r'\|+', '|', x)
                ) #replaces multiple inner | with a single |
            self.db[header] = responses
            self.db.drop(columns=subset,inplace=True)
            
    def convert_surveys(self,surveys,id_field='MRN',
                        time_field='eortc_qlqc30_35_timestamp',
                        time_format='%m/%d/%Y %H:%M'):
        surveys[time_field] = pd.to_datetime(
            surveys[time_field], format=time_format,errors='coerce'
            )
        surveys.insert(1,'time_since_RT_end',None)
        surveys.insert(1,'RT_duration',None)
        surveys['RT_duration'] = surveys[id_field].apply(
            lambda x: None if x not in self.durations.keys() else self.durations[x]
            )
        surveys['time_since_RT_end'] = surveys.apply(
            lambda x: None if x[id_field] not in self.completion_dates.keys() else \
                (x[time_field] - self.completion_dates[x[id_field]]).days,
            axis=1
            )
        return surveys
        

if __name__ == '__main__':
    import sys
    db = pd.read_csv(sys.argv[1])
    test = Database(db)
    test.build_type_map()
    test.clean_data()
    test.handle_grouped_fields()
    test.db.to_csv(sys.argv[2],index=False)