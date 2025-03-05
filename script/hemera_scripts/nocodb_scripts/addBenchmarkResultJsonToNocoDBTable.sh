#!/bin/bash

# Check if directory is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory_with_json_files>"
    exit 1
fi

# Get the directory from the first argument
TARGET_DIR="$1"

# Source the environment
echo "Calling env.sh to set NOCODB env variables"
source ./env.sh

# Iterate over all JSON files in the directory
for json_file in "$TARGET_DIR"/*.json; do
    if [ -f "$json_file" ]; then
        echo "Processing $json_file..."
        ./enter_dataset.py -t ma3j18fd4jwxnls "$json_file"
    else
        echo "No JSON files found in $TARGET_DIR"
    fi
done

