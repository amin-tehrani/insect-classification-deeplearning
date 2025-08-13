#!/bin/bash

#SBATCH -J dl_insect
#SBATCH -n 5
#SBATCH --partition allgroups
#SBATCH --time 20:00:00
#SBATCH --mem 100G
#SBATCH --output output_%j.txt
#SBATCH --error errors_%j.txt
#SBATCH --mail-user t.mohammadamin@gmail.com
#SBATCH --gres=gpu:a40:1


cd $WORKING_DIR
cd insect-classification-deeplearning


srun apptainer run --nv ./runner.sif ./train.py

