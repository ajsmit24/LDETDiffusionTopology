#!/bin/sh
#SBATCH -p et2024
#SBATCH --mem=100G
#SBATCH -t 20:00:00
#SBATCH -n 1
#SBATCH -J MEDdef_j
#SBATCH -o MED_1M_j_%j.out
#SBATCH -e MED_1M_j_%j.err

(cd mpMED_1M; python ../revise_mPlot_v5.py -t j -d mpMED_1M)
