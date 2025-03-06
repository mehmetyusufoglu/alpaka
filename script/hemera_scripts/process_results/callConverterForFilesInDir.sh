#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if python3 is available and version >= 3.8, else load the module
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")
if ! command -v python3 &> /dev/null || [[ "$PYTHON_VERSION" < "3.8" ]]; then
  echo "Python not found or version is too old ($PYTHON_VERSION). Loading python/3.10.4 module..."
  module load python/3.10.4
fi

# Check if a directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <directory_path>"
  exit 1
fi

DIRECTORY="$1"

# Check if the provided argument is a valid directory
if [ ! -d "$DIRECTORY" ]; then
  echo "Error: $DIRECTORY is not a valid directory."
  exit 1
fi

# Iterate over files starting with "babelstream-gpu" or "babelstream-cpu"
found_files=0
for file in "$DIRECTORY"/babelstream-{gpu,cpu}*.txt; do
  if [ -f "$file" ]; then
    found_files=1
    echo "Converting $file..."
    python3 "$SCRIPT_DIR/convertTxtToJson.py" "$file"
  fi
done

if [ "$found_files" -eq 0 ]; then
    echo "No matching files found in $DIRECTORY"
fi

