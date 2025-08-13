#!/bin/bash

#SBATCH -J dl_insect
#SBATCH -n 6
#SBATCH --partition allgroups
#SBATCH --time 20:00:00
#SBATCH --mem 200G
#SBATCH --output output_%j.txt
#SBATCH --error errors_%j.txt
#SBATCH --mail-user t.mohammadamin@gmail.com
#SBATCH --gres=gpu:l40s:6


cd $WORKING_DIR
cd insect-classification-deeplearning



srun apptainer run --nv ./runner.sif -m torch.distributed.run --nproc_per_node=6 ./dnaencoder.py finetune 

