# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 19:34:41 2023

@author: johna

This script is meant to be used AFTER bulk export of DICOM files from the MIM
server. These files will be deposited en masse into a single folder that is
named by the patient ID. This script should be pointed at one of those folders
at a time, with an output pointing to another directory.

Script will migrate DICOM files if they are of the modality set in
"""

import argparse

from contextlib import redirect_stdout

from rpc3dl.files.dicom_filter import Filter

def main():

    parser = argparse.ArgumentParser(
        description='Copy files from source directory to destination directory'
        )
    
    parser.add_argument(
        'src_dir', 
        metavar='source_directory', 
        type=str, 
        help='path to the source directory'
        )
    parser.add_argument(
        'dst_dir', 
        metavar='destination_directory', 
        type=str, 
        help='path to the destination directory'
        )
    parser.add_argument(
        '--support_dir',
        type=str,
        default=None,
        help="Alternate directory to source plan/dose/ss files from"
        )
    parser.add_argument(
        '--modalities', 
        metavar='modalities', 
        type=str, 
        nargs='+', 
        default=[],
        help='Additional modalities to retain, comma separated'
        )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default=None,
        help="Redirect output to this file (creates file if none exists)"
        )
    
    args = parser.parse_args()
    
    print('Source directory:', args.src_dir)
    print('Destination directory:', args.dst_dir)
    
    keep_modality = ["CT","RTPLAN","RTDOSE","RTSTRUCT"]
    
    keep_modality += args.modalities
    
    if args.support_dir is not None:
        print('Support directory:',args.support_dir)
    
    print("Modalities to transfer:",keep_modality)
    
    if args.output is not None:
        with open(args.output,"w") as f:
            with redirect_stdout(f):
                standard_migrate(
                    args.src_dir, 
                    args.dst_dir, 
                    keep_modality,
                    support_dir=args.support_dir
                    )
            f.close()
    else:
        standard_migrate(
            args.src_dir, 
            args.dst_dir, 
            keep_modality,
            support_dir=args.support_dir
            )
        
def standard_migrate(source, dest, modalities,support_dir=None):
    sender = Filter(modalities)
    sender.set_endpoints(source,dest,support=support_dir)
    sender.send_files()
    if support_dir is not None:
        sender.fetch_supportfiles()
    sender.filter_files(location=dest,cleanup=True)
    
if __name__ == '__main__':
    main()