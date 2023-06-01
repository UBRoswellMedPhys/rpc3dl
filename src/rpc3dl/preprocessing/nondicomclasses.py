# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 00:10:48 2023

@author: johna
"""

import os
import pandas as pd

class Condition:
    def __init__(self, category, operator, value):
        self.category = category
        self.operator = operator
        self.value = int(value)
    
    def evaluate(self, row):
        row_value = int(row[self.category])
        if self.operator == "==":
            return row_value == self.value
        elif self.operator == ">":
            return row_value > self.value
        elif self.operator == ">=":
            return row_value >= self.value
        elif self.operator == "<":
            return row_value < self.value
        elif self.operator == "<=":
            return row_value <= self.value
        elif self.operator == "!=":
            return row_value != self.value
        else:
            raise ValueError(f"Unsupported operator: {self.operator}")
    
    def __and__(self, other):
        return CompositeCondition(self, "&", other)
    
    def __or__(self, other):
        return CompositeCondition(self, "|", other)

class CompositeCondition(Condition):
    def __init__(self, left_condition, operator, right_condition):
        self.left_condition = left_condition
        self.operator = operator
        self.right_condition = right_condition

        category = None
        value = None
        if self.operator == "&":
            # When combining with "&", evaluate as False until proven True
            category = f"({self.left_condition.category} & {self.right_condition.category})"
            value = False
        elif self.operator == "|":
            # When combining with "|", evaluate as True until proven False
            category = f"({self.left_condition.category} | {self.right_condition.category})"
            value = True
        else:
            raise ValueError(f"Unsupported operator: {self.operator}")

        super().__init__(category, self.operator, value)

    def evaluate(self, row):
        left_result = self.left_condition.evaluate(row)
        right_result = self.right_condition.evaluate(row)
        if self.operator == "&":
            self.value = left_result and right_result
        elif self.operator == "|":
            self.value = left_result or right_result
        else:
            raise ValueError(f"Unsupported operator: {self.operator}")

        return self.value
    
    def __and__(self, other):
        return CompositeCondition(self, "&", other)
    
    def __or__(self, other):
        return CompositeCondition(self, "|", other)

class Survey:
    def __init__(self,
                 df,
                 time_col = 'eortc_qlqc30_35_timestamp',
                 id_col = 'MRN',
                 days_cutoff=90):
        
        self.data = df.reset_index(drop=True)
        self.cutoff = days_cutoff
        
        if isinstance(id_col,list):
            self.id_col = self._check_list_input(id_col)
        else:
            self.id_col = id_col
        
        if isinstance(time_col, list):
            self.time_col = self._check_list_input(time_col)
        else:
            self.time_col = time_col
            
    def _check_list_input(self,valuelist):
        found = False
        keep = None
        for val in valuelist:
            if val in self.data.columns:
                if found is False:
                    keep = val
                    found = True
                elif found is True:
                    raise ValueError(f"Multiple matches found in {valuelist}")
        return keep
            
    def calculate_time(self,ptinfo):
        
        # clean up dataframes
        self.data[self.time_col] = self.data[self.time_col].apply(
            lambda x: None if str(x).startswith("#") else x
            )
        self.data.drop(
            self.data[self.data[self.time_col] == '[not completed]'].index, 
            inplace=True
            )

        self.data = self.data.dropna(subset=[self.time_col])
        self.data.insert(1,'days_since_RT',None)
        # iterate through rows
        for i,row in self.data.iterrows():
            pt_id = row[self.id_col]
            pt_entries = ptinfo.data[ptinfo.data[ptinfo.id_col]==pt_id]
            if len(pt_entries) == 0:
                # no match in provided ptinfo DB
                continue
            first_RT_done = pt_entries[ptinfo.time_col].min()
            days_since = (
                pd.to_datetime(row[self.time_col]) - pd.to_datetime(first_RT_done)
                ).days
            self.data.loc[i,'days_since_RT'] = days_since
        # drop all rows that never got time data assigned
        prev_len = len(self.data)
        self.data = self.data.dropna(subset=['days_since_RT'])
        new_len = len(self.data)
        if new_len != prev_len:
            print("Note: {} rows dropped due to missing time data.".format(
                prev_len - new_len
                ))
    
    def bin_surveys(self, dest_dir, fields=None):
        if fields is None:
            # use default H&N slate
            fields = [
                self.id_col,
                "overall_health",
                "qol",
                "dry_mouth",
                "sticky_saliva",
                "pain_mouth",
                "pain_jaw",
                "sore_mouth",
                "pain_throat",
                "prob_liquids",
                "prob_pureed",
                "prob_solid",
                "prob_eating",
                "prob_enjoy_meals",
                "lost_wt",
                "gained_wt"
                ]
        new_df = self.data[fields].copy(deep=True)
        new_df.insert(0,"bin","Undefined")
        
        def parse_values(x):
            if x < 0:
                return "acute"
            elif x < self.cutoff:
                return "early"
            elif x >= self.cutoff:
                return "late"
        
        new_df['bin'] = self.data['days_since_RT'].apply(parse_values)
        destpath = os.path.join(
            dest_dir,
            f"{self.cutoff}day_binned_surveys.csv"
            )
        new_df.to_csv(destpath)
    
    def evaluate(self,
                 patient,
                 condition,
                 timeframe,
                 cutoff=90,
                 mode='majority'):
        pt_subset = self.data[self.data[self.id_col]==patient]
        
        # evaluae time windowing and cutoff
        if timeframe == 'acute':
            pt_subset = pt_subset[pt_subset['days_since_RT'] <= 0]
        elif timeframe == 'early':
            pt_subset = pt_subset[
                (pt_subset['days_since_RT'] > 0) & 
                (pt_subset['days_since_RT'] <= cutoff)
                ]
        elif timeframe == 'late':
            pt_subset = pt_subset[pt_subset['days_since_RT'] > cutoff]
        elif timeframe == 'all':
            pass
        else:
            raise Exception("Invalid timeframe argument")
            
        if len(pt_subset) == 0:
            return None
        pos = 0
        neg = 0
        for i,row in pt_subset.iterrows():
            if condition.evaluate(row):
                pos += 1
            else:
                neg += 1
        
        if mode == "majority":
            if pos >= neg:
                return 1
        elif mode == "all":
            if neg == 0:
                return 1
        elif mode == "any":
            if pos > 0:
                return 1
        
        return 0
        
class PatientInfo:
    def __init__(self,
                 df,
                 id_col = 'MRN',
                 time_col = 'RT Completion Date'):
        df = df.rename(columns={col:col.strip() for col in df.columns})
        self.data = df.reset_index(drop=True)
        if isinstance(id_col,list):
            self.id_col = self._check_list_input(id_col)
        else:
            self.id_col = id_col
        
        if isinstance(time_col, list):
            self.time_col = self._check_list_input(time_col)
        else:
            self.time_col = time_col
        
        # unsure about stuff below this for now
        includesRT = self.data['Treatment Type'].astype(str).apply(
            lambda x: "rt" in x.lower()
            )
        # critical to only retain First Diagnosis entries, as RedCap entries
        # do not keep demographic data for subsequent entries
        self.data = self.data[self.data['Event Name']=="First Diagnosis "]
        
        self.data = self.data[includesRT]
        self.data.set_index(self.id_col,inplace=True,drop=False)
        self.encoded = False
        
    def _check_list_input(self,valuelist):
        found = False
        keep = None
        for val in valuelist:
            if val in self.data.columns:
                if found is False:
                    keep = val
                    found = True
                elif found is True:
                    raise ValueError(f"Multiple matches found in {valuelist}")
        return keep
        
    
    def scrub_data(self):
        """
        Store patient data that is important for neural network training
        
        Current working list (caution for trailing spaces):
            Gender
            Race [grouped field]
            Current Smoking Status (within 1 month of treatment)
            Age at Diagnosis ?
            Type [grouped field, refers to disease type] ?
            Disease Site [grouped field]
            T Stage Clinical
            N Stage
            M Stage
            HPV Status
            Treatment Type
            ------------------------
            RT Total Dose to Primary Site (Gy) [not necessarily for training but could be used for verification]
            Height (cm)
            Post Treatment Weight (kg)
            [Date of Last Follow Up] (tough with anonymization, but maybe useful to confirm QOL survey timings?)

        """
        from sklearn.preprocessing import OneHotEncoder
        
        self.scrubbed_data = pd.DataFrame(index=self.data[self.id_col])     
        
        # first take single-column
        self.scrubbed_data.loc[:,'Gender'] = self.data['Gender']
        self.scrubbed_data.loc[:,'Smoking Stats'] = self.data[
            'Current Smoking Status (within 1 month of treatment)'
            ]        
        self.scrubbed_data.loc[:,'T Stage'] = self.data['T Stage Clinical']
        self.scrubbed_data.loc[:,'N Stage'] = self.data['N stage']
        self.scrubbed_data.loc[:,'M Stage'] = self.data['M stage']
        self.scrubbed_data.loc[:,'HPV status'] = self.data['HPV status']
        self.scrubbed_data.loc[:,'Treatment Type'] = self.data['Treatment Type']
        
        self._encoder = OneHotEncoder(sparse_output=False)
        tempdata = self._encoder.fit_transform(self.scrubbed_data)
        self.ohe_data = pd.DataFrame(
            index=self.scrubbed_data.index,
            columns=self._encoder.get_feature_names_out(),
            data=tempdata
            )
        self.ohe_data = self.ohe_data.astype(int)
        
        # anonymization of age requires 90+ be grouped
        self.data['Age at Diagnosis'] = self.data['Age at Diagnosis'].fillna(0)
        self.ohe_data.loc[:,'Age'] = self.data['Age at Diagnosis'].apply(
            lambda x: 90 if int(x) > 90 else int(x)
            )
        
        for groupfield in ['Race','Disease Site']:
            temp = binarize_grouped_field(self.data,groupfield)
            self.ohe_data = pd.concat([self.ohe_data,temp],axis=1)
        # =========
        # possibly add other columns in some other space, or as reference, unsure
        
            
    def to_csv(self,path):
        if not hasattr(self,"scrubbed_data"):
            raise Exception("You must first perform data scrub")
        self.ohe_data.to_csv(path,index=True)
    


def binarize_grouped_field(df,fieldname):
    import re
    fieldex = f"\s*({fieldname})\s*\(choice=(.*)\)\s*"
    subcols = [col for col in df.columns if re.fullmatch(fieldex,col)]
    subdf = df[subcols]
    subdf.fillna("Unchecked",inplace=True)
    for col in subdf.columns:
        subdf.loc[:,col] = subdf[col].apply(
            lambda x: 1 if x=='Checked' else 0
            )
    colrename = {col:re.sub(fieldex,r"\1_\2",col) for col in subdf.columns}
    subdf.rename(mapper=colrename, axis=1, inplace=True)
    return subdf

def resolve_grouped_field(df,fieldname):
    import re
    fieldex = f"\s*{fieldname}\s*\(choice=(.*)\)\s*"
    subcols = [col for col in df.columns if re.fullmatch(fieldex,col)]
    subdf = df[subcols]
    subdf.fillna("Unchecked",inplace=True)
    for col in subdf.columns:
        cat_name = re.sub(fieldex,r"\1",col)
        subdf[col] = subdf[col].apply(
            lambda x: cat_name if x.lower().strip() == 'checked' else ''
        )
    returnseries = subdf.apply(lambda x: "".join(x), axis=1)
    return returnseries