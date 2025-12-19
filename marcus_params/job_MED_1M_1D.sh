#!/bin/sh
#SBATCH -p et2024
#SBATCH --mem=100G
#SBATCH -t 20:00:00
#SBATCH -n 1
#SBATCH -J MEDtrue_1D
#SBATCH -o MED_1M_1D_%j.out
#SBATCH -e MED_1M_1D_%j.err

(cd mpMED_1M; python ../revise_mPlot_v5.py -t 1D -d mpMED_1M)
