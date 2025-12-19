#!/bin/sh
#SBATCH -p et2024
#SBATCH --mem=100G
#SBATCH -t 20:00:00
#SBATCH -n 1
#SBATCH -J BIGdef_j
#SBATCH -o BIG_1B_j_%j.out
#SBATCH -e BIG_1B_j_%j.err

(cd mpBIG_1B; python ../revise_mPlot_v5.py -t j -d mpBIG_1B)
