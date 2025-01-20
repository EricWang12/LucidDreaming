#!/bin/bash
set -e

DEFAULT_CFG="configs/DF-edit/DF.yaml"

# gpu=${1:-$DEFAULT_GPU}
label=${1:-$DEFAULT_CFG}
dir=${2:-"objects/multi_gen/hard"}
# gpu_indices=${3:-(0)}
IFS=' ' read -r -a gpu_indices <<< "$3"
script=${4:-"no_wrapper.sh"}

counter=0
skip_files=${5:-0}
stop_at_file=${6:-20000}

echo "skip_files: $skip_files"
echo "stop_at_file: $stop_at_file"

# gpu_indices=(0 1 2 3)

# if [[ $label != *.yaml ]]; then
#     # If it doesn't, append '.yaml' to the filename
#     label="configs/DF-edit/${label}.yaml"
# fi

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
files=()
cmd=""
for file in "$dir"/*; do
    # echo "$dir"/*
    # Skip the first N files
    ((++counter))
    # echo $counter
    if ((counter <= skip_files)); then
        continue
    fi

    # Stop at the Nth file
    if ((counter == stop_at_file)); then
        echo "in stop!!"
        break
    fi

    # New code starts here

    # cls=$(basename $dir)
    # basename_no_ext="${file##*/}"  # Extract the filename from the path
    # basename_no_ext="${basename_no_ext%.*}"  # Remove the extension from the filename
    # if [[ -d "outputs/multi_gen/$label/$cls/$basename_no_ext" ]]; then  # Check if a directory with the basename exists
    #      echo "Skipping outputs/multi_gen/$label/$cls/$basename_no_ext"
    #     continue  # Skip to the next iteration if the directory exists
    # fi

    if [[ -f "$file" ]]; then
        files+=("$file")
    fi
    
    if [[ ${#files[@]} -eq ${#gpu_indices[@]} ]]; then
        for i in "${!files[@]}"; do

            cmd+="bash scripts/multi_gen/${script} ${gpu_indices[$i]} ${files[$i]} ${label} & "
        done
        
        echo "$cmd"
        eval "$cmd"
        wait 
        sleep 1    
        files=()
        cmd=""
        # break
    fi
done

# exit 0
if [[ ${#files[@]} -gt 0 ]]; then
    # Skip the first N files
    ((++counter))
    if ((counter <= skip_files)); then
        continue
    fi

    # Stop at the Nth file
    if ((counter == stop_at_file)); then
        break
    fi
    
    cmd=""
    for i in "${!files[@]}"; do
        cmd+="bash scripts/multi_gen/${script} ${gpu_indices[$i]} ${files[$i]} ${label} & "
    done

    echo "$cmd"
    eval "$cmd"
    wait
    sleep 1
fi