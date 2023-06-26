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
    get_rois, get_attr_deep
    )
from rpc3dl.preprocessing.arrayclasses import (
    PatientCT, PatientMask, PatientDose
    )
from rpc3dl.preprocessing.handler import Preprocessor

# TODO - will need to update these imports to absolute rpc3dl imports
from gui_objects import ROISelectionPopUp, LabelFileProcess, SimpleTable, ConflictResolver, VisualReview

class RoI:
    # simple class object to hold a few related attributes
    def __init__(self,name=None,ref_num=None):
        self.name = name
        self.ref_num = ref_num
        self.status = False
        
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name

class PreprocessApp(tk.Tk):
    # main app
    def __init__(self):
        # instantiate some initial attributes (blank)
        super().__init__()
        self.FOLDER = " " * 40
        self.DCMFILES = {}
        self._unique_patientIDs = []
        self._unique_frameofref = []
        self.roi = []
        self.label_settings = None
        self.label_file = None
        self.pt_info = None
        
    @property
    def active_patientID(self):
        if len(self._unique_patientIDs) == 1:
            return self._unique_patientIDs[0]
        else:
            return None
        
    @property
    def roi_chosen(self):
        return sum([r.status for r in self.roi])
    
    @property
    def labelfilestatus(self):
        if self.label_settings is not None and self.label_file is not None:
            return "Configured"
        else:
            return "Not configured"
        
    @property
    def file_conflict(self):
        return any((
            len(self._unique_patientIDs) > 1,
            len(self._unique_frameofref) > 1
            ))
    
    def check_files(self):
        self._unique_patientIDs = []
        self._unique_frameofref = []
        for k,v in self.DCMFILES.items():
            for file in v:
                self._unique_patientIDs.append(
                    get_attr_deep(file,"PatientID")
                    )
                self._unique_frameofref.append(
                    get_attr_deep(file,"FrameOfReferenceUID")
                    )
        self._unique_patientIDs = list(set(self._unique_patientIDs))
        self._unique_frameofref = list(set(self._unique_frameofref))
        self.file_conflicts_label.configure(
            text="File conflicts: {}".format(self.file_conflict)
            )
        if self.file_conflict:
            self.resolve_conflicts_button.configure(state=tk.NORMAL)
        
    def calculate_label(self):
        # TODO - create error message popups (message box)
        if self.labelfilestatus != "Configured":
            self.update_label_field("")
            return None
        if len(self._unique_patientIDs) != 1:
            self.update_label_field("")
            return None
        patientID = self._unique_patientIDs[0]
        id_col = None
        for col in self.label_file.columns:
            if patientID in self.label_file[col].values:
                print("ID column found:",col)
                id_col = col
                break
        if id_col is None:
            print("ID not found")
      
        subset = self.label_file[self.label_file[id_col]==patientID]
        if self.label_settings['timebin'] != "All":
            print("Time binning not yet enabled")
            self.update_label_field("")
            return None
        if len(subset) == 0:
            print("No surveys found")
            self.update_label_field("")
            return None
        
        # TODO - there's gotta be a more elegant way to do this
        pos = 0
        neg = 0
        for i, row in subset.iterrows():
            if self.label_settings['condition'].evaluate(row):
                pos += 1
            else:
                neg += 1
        if self.label_settings['method'] == 'Majority':
            if pos >= neg:
                newlabel = 1
        elif self.label_settings['method'] == 'Any':
            if pos > 0:
                newlabel = 1
        elif self.label_settings['method'] == 'All':
            if neg == 0:
                newlabel = 1
        newlabel = 0
        self.update_label_field(newlabel)
        
    def update_label_field(self,newval):
        self.label_entry.delete(0,tk.END)
        self.label_entry.insert(0,str(newval))
        
    
    def select_dir(self):
        # method call on "Browse" button to choose directory housing DICOM files
        dirname = tkfd.askdirectory()
        self.FOLDER = dirname
        display_text = dirname
        # add some spaces for cleaner look in window
        if len(display_text) < 25:
            display_text += " " * (25 - len(display_text))
        self.selected_folder['text'] = display_text
        
    def stage_files(self):
        """Load and stage the files for processing. Loading the files into
        the app allows the user to make some selections and validation prior
        to performing the preprocessing. This is critical for the interactivity
        that is a key piece of this tool.
        """
        self.DCMFILES = {}
        self.roi = []
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
            self._unique_patientIDs.append(
                get_attr_deep(temp,"PatientID")
                )
            self._unique_frameofref.append(
                get_attr_deep(temp,"FrameOfReferenceUID")
                )
        self._unique_patientIDs = list(set(self._unique_patientIDs))
        self._unique_frameofref = list(set(self._unique_frameofref))
        
        self.update_file_status()
        # prepares reference data for user selection of ROIs to include
        if 'RTSTRUCT' in self.DCMFILES.keys():
            if len(self.DCMFILES['RTSTRUCT']) == 1:
                roi_dict = get_rois(self.DCMFILES['RTSTRUCT'][0])
                for roi, ref_num in roi_dict.items():
                    self.roi.append(RoI(roi,ref_num))
        self.check_files()
        
    def update_file_status(self):
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
        self.check_files()
        self.calculate_label()
                    
    def resolve_file_conflicts(self):
        
        if len(self._unique_patientIDs) > 1:
            popup = ConflictResolver(
                self, self.DCMFILES, field="PatientID"
                )
        elif len(self._unique_frameofref) > 1:
            popup = ConflictResolver(
                self,self.DCMFILES,field="FrameOfReferenceUID"
                )
            
            
    
    def open_ROIpopup(self):
        self.roi_popup = ROISelectionPopUp(
            self, self.roi, callback=self.ROIpopup_close
            )
        
    def ROIpopup_close(self):
        self.roi = self.roi_popup.roi_list
        self.display_num_roi['text'] = f"ROI Selected: {self.roi_chosen}"
        
    def labelfile_open(self):
        self.label_file_popup = LabelFileProcess(
            self,self.label_entry
            )
    
    def preview_popup(self):
        self.preview = VisualReview(self,self.handler)
        
    def help_func(self):
        helpstr = """
This is an early prototype of a data preprocessing tool to allow quick and easy
data preparation for deep learning research. This tool takes DICOM files and
converts them into arrays. The key time savings involve automatically handling
the alignment of CT/dose/structure set data and pixel rescaling. The output of
this tool can be used as direct input data for a neural network. Future tools
will help you select the most approrpiate architecture for your research.

Steps:
    - Select directory holding your DICOM files
    - Click "Stage Files" to load the DICOM contents of the selected directory
    - You will then see a summary of the loaded files. If there are any
    conflicts in Patient ID or FrameOfReferenceUID, then you will see "File
    Conflicts" show "True".
    - If there are file conflicts, you can click Resolve Conflicts to select
    which PatientID/FrameOfReferenceUID you'd like to keep staged. Note that
    you must have just one structure set file and just one dose file for this
    tool to work.
    - You may manually set a label for the staged patient. Also, if 
    your research involves endpoints that are not related to the patient
    volume, the tool accepts a "label file" CSV. Click on "Configure Label File"
    to open the view to set this up.
    - Adjust configuration settings to your liking. They are as follows:
        o Pixel Size: this is the pixel height/width in mm. Slice thickness
        is not configurable.
        o Bounding box: this crops the volume to the requested size
            (NOT ENABLED IN PROTOTYPE)
        o Assign label: this lets you manually assign whatever label you wish.
    - When everything is set up, click "Run" to build your array from the
    staged DICOM files.
    - Click "Preview" to open a viewer, which will allow you to check to ensure
    the arrays look how they should.
    - Click "Save" to store the array file as an HDF5 file
"""
        tk.messagebox.showinfo(
            "Help",
            helpstr
            )
        
    def build_arrays(self):
        """Function which builds the actual 4D array from the stored DCMFILES.
        """
        pixel_size = float(self.px_size.get())
        ct_array = PatientCT(self.DCMFILES['CT'])
        ct_array.rescale(pixel_size)
        dose_array = PatientDose(self.DCMFILES['RTDOSE'][0]) # TODO - support multi-file beam pass through
        dose_array.align_with(ct_array)
        mask_arrays = []
        for r in self.roi:
            if r.status == True:
                new_mask = PatientMask(
                    reference=ct_array,
                    ssfile=self.DCMFILES['RTSTRUCT'][0],
                    roi=r.name
                    )
                mask_arrays.append(new_mask)
        handler = Preprocessor(patient_id=self.active_patientID)
        handler.attach(ct_array)
        handler.attach(dose_array)
        handler.attach(mask_arrays)
        self.handler = handler
        
    def save_final(self):
        dest = tkfd.asksaveasfilename(defaultextension=".h5")
        self.handler.save(dest)

        
    def run(self):
        """Main function - this is where we define the geometry of the main
        app screen and route user flow through different steps.
        """
        self.title("RPC3DL Preprocessing Tool")
        self.geometry("800x600+200+200")
        self.columnconfigure(1,minsize=80)
        
        # ==== Row 0: Title Text ====
        tk.Label(self,text="DICOM Preprocessing Tool").grid(row=0,column=0,columnspan=5)
        
        # == Row 1: Folder select for loading DICOM, plus label file status ==
        tk.Label(self,text="Select Folder:").grid(row=1,column=0)
        self.selected_folder = tk.Label(self,text=self.FOLDER,bd=2,bg="#DFDAD9")
        self.selected_folder.grid(row=1,column=1,columnspan=2,padx=5)
        tk.Button(self,text="Browse",command=self.select_dir).grid(row=1,column=3,padx=5)
        tk.Button(self,text="Stage Files",command=self.stage_files).grid(row=1,column=4,padx=5)
        tk.Label(self,text="Label file status:").grid(row=1,column=5,padx=10)
        self.label_file_status_display = tk.Label(
            self,text=self.labelfilestatus,bd=2,bg="#DFDAD9"
            )
        self.label_file_status_display.grid(row=1,column=6)
        tk.Button(self,text="Help",command=self.help_func).grid(row=2,column=5,columnspan=2,pady=10)
        
        # === Row 2: Section break to file info table ===
        tk.Label(self,text="Files Staged").grid(row=2,column=1,columnspan=3,pady=5)
        
        # === Row 3: File breakdown info table, file conflicts notice ===
        self.file_table = SimpleTable(self,rows=1,columns=2)
        self.file_table.grid(row=3,column=1,columnspan=3)
        self.file_conflicthandler = tk.Frame(self)
        self.file_conflicthandler.grid(row=3,column=4,padx=10)
        self.file_conflicts_label = tk.Label(
            self.file_conflicthandler,
            text="File conflicts: {}".format(self.file_conflict)
            )
        self.file_conflicts_label.grid(row=0,column=0)
        self.resolve_conflicts_button = tk.Button(
            self.file_conflicthandler,
            text="Resolve conflicts",
            command=self.resolve_file_conflicts
            )
        self.resolve_conflicts_button.grid(row=1,column=0)
        self.resolve_conflicts_button.configure(state=tk.DISABLED)
        # === Row 4: ROI Selection status and button ===
        tk.Button(self,text="Select ROI",command=self.open_ROIpopup).grid(row=4,column=2,pady=5)
        self.display_num_roi = tk.Label(self,text=f"ROI Selected: {self.roi_chosen}")
        self.display_num_roi.grid(row=4,column=0,columnspan=2)
        
        # === Row 5: Options menu ===
        self.options_menu = tk.Frame(self,bd=5)
        self.options_menu.grid(row=5,column=0,columnspan=5)
        tk.Label(self.options_menu,text="Configuration Settings").grid(
            row=0, column=0, columnspan=3
            )
        # ============================================
        # populate options menu
        # row 1 - pixel size
        tk.Label(self.options_menu,text="Pixel size (mm):").grid(row=1,column=0)
        self.px_size = tk.Entry(self.options_menu)
        self.px_size.insert(0,"1.0")
        self.px_size.grid(row=1,column=1)
        
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
        
        # row 3 - label handling
        tk.Label(self.options_menu,text="Assign label:").grid(row=3,column=0)
        self.label_entry = tk.Entry(self.options_menu,state=tk.NORMAL)
        self.label_entry.grid(row=3,column=1)
        tk.Button(self.options_menu,text="Configure label file",command=self.labelfile_open).grid(row=3,column=2)
        
        
        def validate_options(app):
            # probably want to split this up into multiple subfunctions
            good = True
            # handle bounding box
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
                    good = False
                    print(e)
                    
            # handle pixel size
            try:
                pixel_size = float(self.px_size.get())
                self.px_size.config(bg='white')
                print("Pixel size valid: {}".format(self.px_size.get()))
            except:
                self.px_size.delete(0, tk.END)
                self.px_size.config(bg='red')
                good = False
            
            # handle label
            patient_label = app.label_entry.get()
            self.build_arrays()
            self.preview_button.configure(state=tk.NORMAL)
            self.save_button.configure(state=tk.NORMAL)
            
                
        run_button = tk.Button(
            self.options_menu,
            text="Run",
            command=lambda app=self: validate_options(app)
            )
        run_button.grid(row=4,column=1)
        # ======== END OF OPTIONS MENU ================= 
        self.preview_button = tk.Button(
            self,
            text="Preview",
            command=self.preview_popup
            )
        self.preview_button.grid(row=6,column=2)
        self.preview_button.configure(state=tk.DISABLED)
        
        self.save_button = tk.Button(
            self,
            text='Save',
            command=self.save_final
            )
        self.save_button.grid(row=7,column=2)
        self.save_button.configure(state=tk.DISABLED)
        
        self.mainloop()
            
                
if __name__ == "__main__":       
    app = PreprocessApp()
    app.run()