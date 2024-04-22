#!/bin/sh
#SBATCH -p et3
#SBATCH --mem=14000
#SBATCH -t 7000
#SBATCH -n 1
#SBATCH --output=%x.slurm
#SBATCH --error=%x.err

python pilot_cgValidation.py
