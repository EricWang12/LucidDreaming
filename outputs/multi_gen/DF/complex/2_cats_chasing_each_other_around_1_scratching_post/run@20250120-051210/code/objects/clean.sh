#!/bin/bash

# Replace with your directory
DIRECTORY=objects/multi_gen/complex

# Find and remove all empty files in the specified directory and its subdirectories
find "$DIRECTORY" -type f -empty -delete

echo "All empty files have been deleted."
