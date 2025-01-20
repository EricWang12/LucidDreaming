#!/bin/bash

DEFAULT_CFG="DF"
DEFAULT_GPU=0
gpu=${1:-$DEFAULT_GPU}
label=${2:-$DEFAULT_CFG}
dir=${3:-"objects/multi_gen"}

# Check if the directory name is provided
if [[ -z "$dir" ]]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Check if the directory exists
if [[ ! -d "$dir" ]]; then
    echo "Error: $dir is not a directory."
    exit 1
fi

# Iterate over all files in the specified directory
for file in "$dir"/*; do
    # Check if the item is a file (and not a directory)
    if [[ -f "$file" ]]; then
        # Run run.sh with the file name as an argument
        # echo  $gpu $label $file
        # bash scripts/multi_gen/run.sh $gpu $label $file
        bash scripts/multi_gen/no_wrapper.sh $gpu $file

    fi
done
