# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 02:04:13 2023

@author: johna
"""

import tkinter as tk
from tkinter import ttk

       
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