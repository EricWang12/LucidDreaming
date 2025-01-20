#!/bin/bash
set -e
# Declare an array with the list values
declare -a list=("DF" "PD-p" "magic3d")

# This script will pair each GPU index (0 to 3) with a value from the list,
# and execute all.sh in the background with these arguments.

for i in {0..2}
do
    bash scripts/vanilla/all.sh $i ${list[$i]} &
done

# Wait for all background processes to finish
wait