#!/bin/bash

PYTHON_EXECUTABLE="python3"

# Set the path to your prompting.py script
SCRIPT_PATH="prompting.py"

# Set the input CSV file, input language, output language, and prompting type
INPUT_CSV="data/valid_cleaned_parallel_sentences.txt"
INPUT_LANG="cantonese"
OUTPUT_LANG="mandarin"
PROMPTING_TYPE="few_shot"  # Options: "zero" or "few_shot"

# Run the Python script
$PYTHON_EXECUTABLE $SCRIPT_PATH \
    --input_csv $INPUT_CSV \
    --input_lang $INPUT_LANG \
    --output_lang $OUTPUT_LANG \
    --prompting_type $PROMPTING_TYPE