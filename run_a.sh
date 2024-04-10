#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate orchloc

# Set the base directory to the directory of the script
BASE_DIR=$(dirname "$0")

# Construct the absolute path for area_1.py
py_0="$BASE_DIR/classifier_area_a.py"


# Example usage without specifying a configuration file: python "$py_0"
# Example usage with specifying a configuration file: python "$py_0" --config area_a.yml

python "$py_0" --config area_a.yml



