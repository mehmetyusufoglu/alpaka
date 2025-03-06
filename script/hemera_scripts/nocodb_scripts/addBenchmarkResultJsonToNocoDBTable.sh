#!/bin/bash

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if python3 is available and version >= 3.8, else load the module
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")
if ! command -v python3 &> /dev/null || [[ "$PYTHON_VERSION" < "3.8" ]]; then
  echo "Python not found or version is too old ($PYTHON_VERSION). Loading python/3.10.4 module..."
  module load python/3.10.4
fi

# Check if directory is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory_with_json_files>"
    exit 1
fi

TARGET_DIR="$1"

# Source the environment from the same directory
echo "Calling env.sh to set NOCODB env variables"
source "$SCRIPT_DIR/env.sh"

# Ensure Python can find local modules
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

found_files=0
for json_file in "$TARGET_DIR"/*.json; do
    if [ -f "$json_file" ]; then
        found_files=1
        echo "Processing $json_file..."
        python3 "$SCRIPT_DIR/enter_dataset.py" -t ma3j18fd4jwxnls "$json_file"
    fi
done

if [ "$found_files" -eq 0 ]; then
    echo "No JSON files found in $TARGET_DIR"
fi

