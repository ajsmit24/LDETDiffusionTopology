#!/bin/sh
#SBATCH -p et2024
#SBATCH --mem=100G
#SBATCH -t 20:00:00
#SBATCH -n 1
#SBATCH -J MEDeff_3D
#SBATCH -o MED_1M_3D_%j.out
#SBATCH -e MED_1M_3D_%j.err

(cd mpMED_1M; python ../revise_mPlot_v5.py -t 3D -d mpMED_1M)
