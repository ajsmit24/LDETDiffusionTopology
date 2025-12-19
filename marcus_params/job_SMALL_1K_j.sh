#!/bin/sh
#SBATCH -p et2024
#SBATCH --mem=100G
#SBATCH -t 20:00:00
#SBATCH -n 1
#SBATCH -J SMALLdef_j
#SBATCH -o SMALL_1K_j_%j.out
#SBATCH -e SMALL_1K_j_%j.err

(cd mpSMALL_1K; python ../revise_mPlot_v5.py -t j -d mpSMALL_1K)
