#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate orchloc

# Set the base directory to the directory of the script
BASE_DIR=$(dirname "$0")

# rm -r "$BASE_DIR/output"

py_0="$BASE_DIR/0_location_representation.py"

py_1="$BASE_DIR/1_pretrain_cgm.py"

py_2="$BASE_DIR/2_finetune_cgm.py"

py_3="$BASE_DIR/3_generate_csi_database.py"

py_4="$BASE_DIR/classifier_area_b.py"


# Example usage without specifying a configuration file: python "$py_0"
# Example usage with specifying a configuration file: python "$py_0" --config area_b.yml


echo -e "\n##### Running the pipeline for Area B #####\n"


echo -e "\n##### Running 0_location_representation.py --config area_b.yml #####\n"
python "$py_0" --config area_b.yml
echo -e "\n##### finished 0_location_representation.py --config area_b.yml #####\n\n"
sleep 5


echo -e "\n##### Running 1_pretrain_cgm.py --config area_b.yml #####\n"
python "$py_1" --config area_b.yml
echo -e "\n##### finished 1_pretrain_cgm.py --config area_b.yml #####\n\n"
sleep 5


echo -e "\n##### Running 2_finetune_cgm.py --config area_b.yml #####\n"
python "$py_2" --config area_b.yml
echo -e "\n##### finished 2_finetune_cgm.py --config area_b.yml #####\n\n"
sleep 5


echo -e "\n##### Running 3_generate_csi_database.py --config area_b.yml #####\n"
python "$py_3" --config area_b.yml
echo -e "\n##### finished 3_generate_csi_database.py --config area_b.yml #####\n\n"
sleep 5


# echo -e "\n##### Running classifier_area_b.py --config area_b.yml #####\n"
# python "$py_4" --config area_b.yml
# echo -e "\n##### finished classifier_area_b.py --config area_b.yml #####\n\n"
# sleep 5

echo -e "\n##### Finished running the pipeline for Area B #####\n\n"


