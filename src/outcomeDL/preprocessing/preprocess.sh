#!/bin/bash

# Define the path to the list of directory names
dir_list="directories.txt"

# Define the path to the source parent directory
source_dir="/path/to/source/folder"

# Define the path to the destination folder for output files
output_dir="/path/to/output/folder"

# Define the path to the error log file
error_log="/path/to/error.log"

# Loop over the directories in the directory list
while read directory; do
    # Construct the absolute path to the directory
    directory_path="$source_dir/$directory"

    # Construct the name for the output file
    output_file="$output_dir/$directory.h5"

    # Run the Python script with the appropriate arguments
    python run.py "$directory_path" "$output_file" -px 2 -pl -pr -bb "40,128,128" -c || echo "$directory" >> "$error_log"

done < "$dir_list"
