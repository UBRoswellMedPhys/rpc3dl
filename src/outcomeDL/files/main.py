import dicom

import os
import shutil

source_folder = "/home/johnasbach/Research/dcm_storage"
dest_folder = "/home/johnasbach/Research/cleaned_dcm"

for item in os.listdir(source_folder):
    print(item)
    subdirpath = os.path.join(source_folder,item)
    if not os.path.isdir(subdirpath):
        continue
    files = os.listdir(subdirpath)
    if len(files) == 0:
        print("{} has no files.".format(item))
        continue
    dest_subpath = os.path.join(dest_folder,item)
    if not os.path.exists(dest_subpath):
        os.mkdir(dest_subpath)
    success = dicom.main_filter(
        subdirpath,dest_subpath,modalitykeep=['CT','RTDOSE','RTSTRUCT','RTPLAN']
        )
    if success:
        shutil.rmtree(subdirpath,ignore_errors=True)
    