#!/bin/bash
set -e

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
        base_name=$(basename "$file" | sed 's/\(.*\)\..*/\1/')
        # Replace underscores with spaces
        formatted_name=$(echo "$base_name" | tr '_' ' ')

        bash scripts/vanilla/run.sh $gpu $label "${formatted_name}"
        # bash scripts/vanilla/run.sh $gpu $label "$formatted_name"
    fi
done
