# Sequence of work  

* Data to MIM  
    * Requires: no tools  
    * Status: Complete  
  
* Export MIM to Linux computer  
    * Requires: DCMTK tools, bash script.  
    * Status: Scripts are written for query and retrieve, but some exports had issues.  
  
* Identify patients whose exports didn't work right for further investigation.  
    * Requires: script to check destination folders and identify patients whose data export failed so that I can revisit the previous step for those patients.  
    * Status: Not built  
    * BLOCKER: I need to reference what this failure looks like (i.e. do no files come do some files come but not all, etc) so that I can know how to write this.  

* Clean/filter data (study select)  
    * Requires: Python script, DICOM files  
    * Status: Ready, waiting on export  
        outcomeDL/files/dicom_filter.py -> Filter class facilitates this  
        outcomeDL/files/migrate_cli.py -> performs the operation  
  
* Prepare labels and patient info  
    * Requires: Python script, RedCap exports  
    * Status: Ready, waiting on Linux computer  
        outcomeDL/files/generate_labels.py -> prepares labels and stores patient attributes  
        (not built for CLI use, file is meant to be opened, variables edited, and run)  
  
* Anonymizer (optional)  
    * Requires: Python script, DICOM files, supplemental CSVs  
    * Status: Ready but untested  
        outcomeDL/anon/anon.py -> built for CLI use, mass anonymizes DICOM/CSV files  
        Notes: This DOES NOT anonymize the CSV files other than the MRN -> AnonID transform  
  
* Prepare arrays (CT/dose/mask)  
    * Requires: Python script, filtered DICOM files  
    * Status: Early draft ready, functional, could use a re-write  
        outcomeDL/preprocessing/extract_arrays.py -> processes data into arrays  
        Notes: not built for CLI use, open file, edit variables, and run. Would be good to build this such that we can easily change what organ the focus is on
  
* Model training  
    * Requires: Python script, arrays, supplemental CSVs  
    * Status: Early draft ready, functional, could use a re-write  
        outcomeDL/training/main.py -> script to manage training process  
        Notes: Built to pair with a .cfg file to set up training run settings,  
        can be run CLI with this type of file  
        Working on transition to use tf.Dataset (from generator) to feed training data
  
* Different model structures (optional)  
    * Requires: Python script  
    * Status: partially built  
        outcomeDL/training/build_model.py -> import module to call specific model types  
        Notes: needs expansion. would like to look at transformers at the very least  
  
* Evaluation scripts  
    * Requires: Python script  
    * Status: not written  
        Notes: Would like to fully automate model evalution to facilitate multiple runs easily  

* NN activation mapping (model interpretability)  
    * Requires: trained model, Python script  
    * Status: not written  
        Notes: this is the part I'm least clear on, I need to research better how to do this  