#!/bin/bash
# use doubleTranslator2 single prompt for 直译意译
# Check if filename parameter exists
if [ -z "$1" ]; then
    echo "Error: Please provide a filename as a command line argument"
    exit 1
fi

# Get the filename from command line argument
filename=$1

# Remove the extension from the filename
filename_no_ext="${filename%.*}"

output_filename="${filename_no_ext}_rawtxt.md"

echo $output_filename
# Call the Python script with the filename parameter
python3 pdf_translator.py "$filename"

python3 doubleTranslator2.py "$output_filename"