# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 02:04:13 2023

@author: johna
"""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as tkfd

import pandas as pd

from rpc3dl.preprocessing.nondicomclasses import Condition, CompositeCondition
from rpc3dl.files._dicom_util import get_attr_deep, organize_list

class SimpleTable(tk.Frame):
    # reference - https://stackoverflow.com/questions/11047803/creating-a-table-look-a-like-tkinter/11049650#11049650
    def __init__(self, parent, rows=10, columns=2):
        # use black background so it "peeks through" to 
        # form grid lines
        tk.Frame.__init__(self, parent, background="black")
        self._widgets = []
        for row in range(rows):
            current_row = []
            for column in range(columns):
                label = tk.Label(self, text=" ", 
                                 borderwidth=0, width=10)
                label.grid(row=row, column=column, sticky="nsew", padx=1, pady=1)
                current_row.append(label)
            self._widgets.append(current_row)

        for column in range(columns):
            self.grid_columnconfigure(column, weight=1)


    def set(self, row, column, value):
        widget = self._widgets[row][column]
        widget.configure(text=value)
        
class ConflictResolver(tk.Toplevel):
    def __init__(self,master,DCMFILES,field=None):
        super().__init__(master)
        self.title("Conflict Resolver: {}".format(field))
        self.transient(master)
        self.parent_app = master
        
        flatDCMFILES = []
        for k,v in DCMFILES.items():
            flatDCMFILES += v
        
        # prep a nested dictionary
        self.sorted_files = organize_list(flatDCMFILES,field)
        for k,v in self.sorted_files.items():
            self.sorted_files[k] = organize_list(v,"Modality")
        options = list(self.sorted_files.keys())
        self.selected_val = tk.StringVar(self)
        tk.OptionMenu(self,self.selected_val,*options).pack(pady=10)
        tk.Button(self,text="Save",command=self.save).pack(pady=5)
        self.file_display = SimpleTable(self,rows=1,columns=2)
        self.file_display.pack()
        self.selected_val.set(options[0])
        self.selected_val.trace("w",self.update_table)
        
        
    def update_table(self, *args):
        selection = self.selected_val.get()
        self.file_display.pack_forget()
        self.file_display = SimpleTable(
            self,
            rows=len(self.sorted_files[selection]),
            columns=2
            )
        for i, (k,v) in enumerate(self.sorted_files[selection].items()):
            self.file_display.set(i,0,k)
            self.file_display.set(i,1,len(v))
        self.file_display.pack()
        
    def save(self):
        selection = self.selected_val.get()
        self.parent_app.DCMFILES = self.sorted_files[selection]
        self.parent_app.update_file_status()
        
        self.destroy()
        
       
class ROISelectionPopUp(tk.Toplevel):
    def __init__(self,master,roilist,callback=None):
        super().__init__(master)
        self.roi_list = roilist
        self.callback = callback
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        for i, r in enumerate(self.roi_list):
            checkbox_var = tk.BooleanVar(value=r.status)
            checkbox = tk.Checkbutton(
                self, text=r.name, variable=checkbox_var,
                onvalue=True, offvalue=False,
                command=lambda var=checkbox_var, index=i: self.toggle_checkbox(var, index))
            checkbox.pack(anchor="w")
            
    def toggle_checkbox(self, var, index):
        # function for handling checkbox ticking in ROI selection
        self.roi_list[index].status = var.get()
        
    def on_close(self):
        if self.callback is not None:
            self.callback()
        self.destroy()
        
class LabelFileProcess(tk.Toplevel):
    def __init__(self,master,label_entry_field):
        super().__init__(master)
        self.title("Label File Configuration")
        self.transient(master)
        self.parent_app = master
        self.conditions = []
        self.label_entry = label_entry_field
        self.columns = []
        
        tk.Button(
            self,
            text="Attach Label CSV",
            command=lambda x=master: self.select_lblfile(x)
            ).grid(row=0,column=0,pady=10,padx=10,columnspan=2)
        tk.Button(
            self,
            text="Attach Patient Info File",
            command=lambda x=master: self.select_ptinfofile(x)
            ).grid(row=0,column=2,pady=10,padx=10,columnspan=2)
        tk.Label(
            self,text="Select Timeframe",
            borderwidth=1, relief="solid",
            padx=5, pady=5,
            highlightthickness=1, highlightbackground="gray"
            ).grid(row=1,column=0,columnspan=2)
        tk.Label(
            self,text="Method for Multiple Entries",
            borderwidth=1, relief="solid",
            padx=5, pady=5,
            highlightthickness=1, highlightbackground="gray"
            ).grid(row=1,column=2,columnspan=2)
        time_options = ["All", "Early", "Late"]
        self.timebin = tk.StringVar(self)
        self.timebin.set(time_options[0])  # Set the initial selected option
        self.timedropdown = tk.OptionMenu(
            self, self.timebin, *time_options
            )
        self.timedropdown.grid(row=2,column=0,columnspan=2)
        self.timedropdown.configure(state=tk.DISABLED)
        tk.Label(self,text="(Requires Patient Info File)").grid(row=3,column=0,columnspan=2)
        method_options = ['Majority','Any','All'] # TODO - add things like mean
        self.multimethod = tk.StringVar(self)
        self.multimethod.set(method_options[0])
        methoddropdown = tk.OptionMenu(
            self, self.multimethod, *method_options
            )
        methoddropdown.grid(row=2,column=2,columnspan=2)
        
        self.condition_frame = tk.Frame(self,bg="#DFDAD9")
        self.condition_frame.grid(row=4,column=0,columnspan=4,pady=10)
        self.active_condition_index = -1
        self.condition_categories = []
        self.condition_operators = []
        self.condition_values = []
        
        
        
        self.add_row_button = tk.Button(
            self,
            text="Add Condition Row",
            command=self.add_condition_row
            )
        self.add_row_button.grid(row=5,column=0,columnspan=4)
        self.add_row_button.configure(state=tk.DISABLED)
        
        
        
        tk.Button(self,text="Save",command=self.save).grid(row=6,column=0,columnspan=4,pady=10)
        
        
    def select_lblfile(self,master):
        filepath = tkfd.askopenfilename()
        #attaches CSV to the app
        master.label_file = pd.read_csv(filepath)
        self.columns = master.label_file.columns
        tk.Label(self.condition_frame,text="Field").grid(row=0,column=0)
        tk.Label(self.condition_frame,text="Value").grid(row=0,column=2)
        self.add_condition_row()
        self.add_row_button.configure(state=tk.NORMAL)
        self.grab_set()
        
    def select_ptinfofile(self,master):
        filepath = tkfd.askopenfilename()
        master.pt_info = pd.read_csv(filepath)
        self.timedropdown.configure(state=tk.NORMAL)
        
    def store_active_condition(self):
        category = self.condition_categories[self.active_condition_index].get()
        operator = self.condition_operators[self.active_condition_index].get()
        value = self.condition_frame.grid_slaves(row=self.active_condition_index,column=2)[0].get()
        for col in range(3):
            self.condition_frame.grid_slaves(row=self.active_condition_index,column=col)[0].configure(state=tk.DISABLED)
        cond = Condition(category, operator, float(value))
        self.conditions.append(cond)
        
        
    def add_condition_row(self):
        if self.active_condition_index >= 0:
            # skips storage for first row instantiation
            self.store_active_condition()
        self.active_condition_index += 1
        newvar = tk.StringVar(self)
        newvar.set("-")
        self.condition_categories.append(newvar)
        tk.OptionMenu(
            self.condition_frame,
            self.condition_categories[self.active_condition_index],
            "-",
            *self.columns
            ).grid(row=self.active_condition_index,column=0)
        operators = [">","<","==",">=","<=","!="]
        newop = tk.StringVar(self)
        newop.set("==")
        self.condition_operators.append(newop)
        tk.OptionMenu(
            self.condition_frame,
            self.condition_operators[self.active_condition_index],
            "==",
            *operators
            ).grid(row=self.active_condition_index,column=1)
        value_field = tk.Entry(self.condition_frame)
        value_field.grid(row=self.active_condition_index,column=2)
        
    
    def save(self):
        self.store_active_condition()
        fullconfig = {}
        fullconfig['timebin'] = self.timebin.get()
        fullcondition = self.conditions[0]
        i = 1
        while i < len(self.conditions):
            fullcondition = fullcondition & self.conditions[i]
            i += 1
        fullconfig['condition'] = fullcondition
        fullconfig['method'] = self.multimethod.get()
        self.parent_app.label_settings = fullconfig
        self.parent_app.label_file_status_display['text'] = \
            self.parent_app.labelfilestatus
        self.parent_app.calculate_label()
        self.destroy()