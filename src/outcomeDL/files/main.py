import _dicom_util as dicom
import pydicom

import os
import shutil

source_folder = "/home/johnasbach/Research/dcm_storage"
dest_folder = "/home/johnasbach/Research/cleaned_dcm"

def clean_bulk_export(source_folder,dest_folder):
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
        elif not success:
            os.rmdir(dest_subpath)

def select_RS(parent_dir):
    """Assumes file structure of MRN -> StudyUID -> files
    """
    patient_folders = [
        os.path.join(parent_dir,subdir) for subdir in os.listdir(parent_dir)
        ]
    for patient in patient_folders:
        print("Processing {}".format(patient)) 
        studies = [
            os.path.join(patient,studydir) for studydir in os.listdir(patient)
        ]
        if len(studies) > 1:
            print("More than one study")
        for study in studies: 
            files = [ 
                os.path.join(study,file) for file in os.listdir(study)
            ]
            ss_files = [file for file in files if file.split('/')[-1].startswith("RS")]
            if len(ss_files) == 1:
                ss = pydicom.dcmread(ss_files[0])
                if ss.ApprovalStatus != "APPROVED":
                    print("No approved RS files.")
                continue
            # if more than one, load them all to check
            loaded_ss = [pydicom.dcmread(sspath) for sspath in ss_files]
            to_save = []
            for path, ss in zip(ss_files,loaded_ss):
                if str(ss.ApprovalStatus).upper() == "APPROVED":
                    to_save.append(path)
            if len(to_save) == 0:
                print("No approved RS files")
                continue
            else:
                print("Retaining {} RS files".format(len(to_save)))
            for path in ss_files:
                if path not in to_save:
                    os.remove(path)
                    

if __name__ == "__main__":
    select_RS(dest_folder)