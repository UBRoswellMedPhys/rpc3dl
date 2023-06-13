# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 16:33:48 2023

@author: johna
"""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as tkfd
from PIL import Image, ImageTk

import os
import pydicom
from pydicom.errors import InvalidDicomError
from rpc3dl.files._dicom_util import (
    get_rois
    )

# TODO - will need to update these imports to absolute rpc3dl imports
from popups import ROISelectionPopUp

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

class RoI:
    def __init__(self,name=None,ref_num=None):
        self.name = name
        self.ref_num = ref_num
        self.status = False
        
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name

class PreprocessApp(tk.Tk):
    
    def __init__(self):
        # instantiate some initial attributes (blank)
        super().__init__()
        self.FOLDER = " " * 20
        self.DCMFILES = {}
        self.roi = []
        self.configuration = {}
        
    @property
    def roi_chosen(self):
        return sum([r.status for r in self.roi])
    
    def select_dir(self):
        # method call on "Browse" button to choose directory housing DICOM files
        dirname = tkfd.askdirectory()
        self.FOLDER = dirname
        self.selected_folder['text'] = self.FOLDER
        
    def stage_files(self):
        """Load and stage the files for processing. Loading the files into
        the app allows the user to make some selections and validation prior
        to performing the preprocessing. This is critical for the interactivity
        that is a key piece of this tool.
        """
        self.DCMFILES = {}
        filepaths = [
            os.path.join(self.FOLDER,file) for file in os.listdir(self.FOLDER)
            ]
        filepaths = [p for p in filepaths if os.path.isfile(p)]
        for p in filepaths:
            try:
                temp = pydicom.dcmread(p)
            except InvalidDicomError:
                # bypass non-DICOM files
                continue
            if temp.Modality not in self.DCMFILES:
                self.DCMFILES[temp.Modality] = []
            self.DCMFILES[temp.Modality].append(temp)
        # build table - allows user to see what's been staged
        self.file_table.grid_remove()
        self.file_table = SimpleTable(
            self,rows=len(self.DCMFILES),columns=2
            )
        self.file_table.grid(row=3,column=1,columnspan=3)
        # populates table
        for i, (k,v) in enumerate(self.DCMFILES.items()):
            self.file_table.set(i,0,k)
            self.file_table.set(i,1,len(v))
        # prepares reference data for user selection of ROIs to include
        if 'RTSTRUCT' in self.DCMFILES.keys():
            if len(self.DCMFILES['RTSTRUCT']) == 1:
                roi_dict = get_rois(self.DCMFILES['RTSTRUCT'][0])
                for roi, ref_num in roi_dict.items():
                    self.roi.append(RoI(roi,ref_num))
    
    def open_popup(self):
        self.roi_popup = ROISelectionPopUp(
            self, self.roi, callback=self.popup_close
            )
        
    def popup_close(self):
        self.roi = self.roi_popup.roi_list
        self.display_num_roi['text'] = f"ROI Selected: {self.roi_chosen}"

    # def toggle_checkbox(self, var, index):
    #     # function for handling checkbox ticking in ROI selection
    #     self.roi[index].status = var.get()

    # def open_popup(self):
    #     popup = tk.Toplevel(self)
    #     popup.title("ROI Selection")
    #     popup.minsize(200,200)

    #     # Create checkboxes in the popup window
    #     for i, r in enumerate(self.roi):
    #         checkbox_var = tk.BooleanVar(value=r.status)
    #         checkbox = tk.Checkbutton(
    #             popup, text=r.name, variable=checkbox_var,
    #             onvalue=True, offvalue=False,
    #             command=lambda var=checkbox_var, index=i: self.toggle_checkbox(var, index))
    #         checkbox.pack(anchor="w")

    #     # Add a confirmation button in the popup window
    #     confirm_button = tk.Button(
    #         popup, 
    #         text="Confirm", 
    #         command=lambda: self.handle_popup_selection(popup))
    #     confirm_button.pack()

    # def handle_popup_selection(self, popup):
    #     # Update display
    #     # BUG - if user toggles checkboxes but then EXITS the popup window,
    #     # the status of ROIs will still be updated but display will not update
    #     self.display_num_roi['text'] = f"ROI Selected: {self.roi_chosen}"
    #     # Close the popup window
    #     popup.destroy()

        
    def run(self):
        self.title("RPC3DL Preprocessing Tool")
        self.geometry("600x520+50+50")
        self.columnconfigure(1,minsize=80)
        tk.Label(self,text="DICOM Preprocessing Tool").grid(row=0,column=0,columnspan=5)
        tk.Label(self,text="Select Folder:").grid(row=1,column=0)
        self.selected_folder = tk.Label(self,text=self.FOLDER,bd=2,bg="#DFDAD9")
        self.selected_folder.grid(row=1,column=1,columnspan=2,padx=5)
        
        tk.Button(self,text="Browse",command=self.select_dir).grid(row=1,column=3,padx=5)
        tk.Button(self,text="Stage Files",command=self.stage_files).grid(row=1,column=4,padx=5)
        tk.Label(self,text="Files Staged").grid(row=2,column=1,columnspan=3,pady=10)
        self.file_table = SimpleTable(self,rows=1,columns=2)
        self.file_table.grid(row=3,column=1,columnspan=3)
        tk.Button(self,text="Select ROI",command=self.open_popup).grid(row=4,column=2,pady=5)
        self.display_num_roi = tk.Label(self,text=f"ROI Selected: {self.roi_chosen}")
        self.display_num_roi.grid(row=4,column=0,columnspan=2)
        self.options_menu = tk.Frame(self,bd=5)
        self.options_menu.grid(row=5,column=0,columnspan=5)
        tk.Label(self.options_menu,text="Configuration Settings").grid(
            row=0, column=0, columnspan=3
            )
        
        # populate options menu
        # row 1 - pixel size
        tk.Label(self.options_menu,text="Pixel size (mm):").grid(row=1,column=0)
        px_size = tk.Entry(self.options_menu)
        px_size.insert(0,"1.0")
        px_size.grid(row=1,column=1)
        
        # row 2 - bounding box
        bounding_box_bool = tk.BooleanVar()
    
        bblabel = tk.Label(self.options_menu,text="(slices x width x height)")
        bbentry = tk.Entry(self.options_menu,state=tk.DISABLED)
        def toggle_entry():
            entry_state = bounding_box_bool.get()
            bbentry.config(state=tk.NORMAL if entry_state else tk.DISABLED)
            
        bounding_box_check = tk.Checkbutton(
            self.options_menu,
            text="Bounding Box",
            variable=bounding_box_bool,
            command=toggle_entry
            )
        bounding_box_check.grid(row=2,column=0)
        bblabel.grid(row=2,column=2)
        bbentry.grid(row=2,column=1)
        
        def validate_options():
            if bounding_box_bool.get():
                bb = bbentry.get()
                # try both delimiters, x and comma
                bb = bb.split("x")
                if len(bb) == 1:
                    bb = bb[0].split(",")
                try:
                    bb = [int(d.strip()) for d in bb]
                    assert len(bb) == 3
                    bbentry.config(bg='white')
                    print("Bounding box valid: {}".format(bb))
                except Exception as e:
                    bbentry.delete(0,tk.END)
                    bbentry.config(bg="red")
                    print(e)
            
            try:
                float(px_size.get())
                px_size.config(bg='white')
                print("Pixel size valid: {}".format(px_size.get()))
            except:
                px_size.delete(0, tk.END)
                px_size.config(bg='red')
                
        validate_button = tk.Button(
            self.options_menu,
            text="Run",
            command=validate_options
            )
        validate_button.grid(row=3,column=1)
                
        
        self.mainloop()
            
                
if __name__ == "__main__":       
    app = PreprocessApp()
    app.run()