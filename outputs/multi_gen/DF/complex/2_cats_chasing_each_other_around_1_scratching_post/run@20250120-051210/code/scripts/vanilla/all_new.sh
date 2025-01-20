#!/bin/bash
set -e

# Trap SIGINT and send SIGTERM to the process group
trap 'killall=1' SIGINT


DEFAULT_CFG="DF"
DEFAULT_GPU=0
# gpu=${1:-$DEFAULT_GPU}
label=${1:-$DEFAULT_CFG}
dir=${2:-"objects/multi_gen"}
IFS=' ' read -r -a gpu_indices <<< "$3"


# script=${4:-"no_wrapper.sh"}
# gpu_indices=(2 2)

skip_files=${4:-0}
stop_at_file=${5:-20000}
counter=0


cls=$(basename $dir)

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

    # Skip the first N files
    ((++counter))
    if ((counter <= skip_files)); then
        continue
    fi

    # Stop at the Nth file
    if ((counter == stop_at_file)); then
        break
    fi

    base_name=$(basename $file | sed 's/\(.*\)\..*/\1/')
    if  [[ -d "outputs/vanilla_generation/$label/$cls/$base_name" ]]; then
        echo "Skipping $base_name"
        continue
    fi

    if [[ -f "$file" ]]; then
        files+=("$file")
    fi


    if [[ ${#files[@]} -eq ${#gpu_indices[@]} ]]; then
        for i in "${!files[@]}"; do
            base_name=$(basename "${files[$i]}" | sed 's/\(.*\)\..*/\1/')
            # Replace underscores with spaces
            formatted_name=$(echo "$base_name" | tr '_' ' ')


            # bash scripts/vanilla/run.sh $gpu $label "${formatted_name}"
            cmd+="bash scripts/vanilla/run.sh  ${gpu_indices[$i]} ${label} \"${formatted_name}\" ${cls} & "
        done
        
        # echo "$cmd"
        eval "$cmd"
        wait 
        sleep 1    
        files=()
        cmd=""
    fi
done

if [[ ${#files[@]} -gt 0 ]]; then

    ((++counter))
    if ((counter <= skip_files)); then
        continue
    fi
    
    if ((counter == stop_at_file)); then
        break
    fi

    cmd=""
    for i in "${!files[@]}"; do

        base_name=$(basename "${files[$i]}" | sed 's/\(.*\)\..*/\1/')
        # Replace underscores with spaces
        formatted_name=$(echo "$base_name" | tr '_' ' ')

        # bash scripts/vanilla/run.sh $gpu $label "${formatted_name}"
        cmd+="bash scripts/vanilla/run.sh  ${gpu_indices[$i]} ${label} \"${formatted_name}\" ${cls} & "
        # cmd+="bash scripts/multi_gen/no_wrapper_global.sh ${gpu_indices[$i]} ${files[$i]} ${label} & "
    done

    # echo "$cmd"
    eval "$cmd"
    wait
    sleep 1
fi