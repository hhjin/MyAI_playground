#!/bin/bash

# Check if filename parameter exists
if [ -z "$1" ]; then
    echo "Error: Please provide a filename as a command line argument"
    exit 1
fi

# Get the filename from command line argument
filename=$1

# Remove the extension from the filename
filename_no_ext="${filename%.*}"

output_filename="${filename_no_ext}_百度翻译.md"

echo $output_filename
# Call the Python script with the filename parameter
python3 pdf_translator.py "$filename"  --do_translate

python3 chinesePolish.py "$output_filename"