#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate orchloc

# Set the base directory to the directory of the script
BASE_DIR=$(dirname "$0")

py_4="$BASE_DIR/classifier_area_b.py"


# Example usage without specifying a configuration file: python "$py_0"
# Example usage with specifying a configuration file: python "$py_0" --config area_b.yml


python "$py_4" --config area_b.yml


