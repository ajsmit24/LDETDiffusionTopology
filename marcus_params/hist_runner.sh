#!/bin/sh
#SBATCH -p et2024
#SBATCH --mem=100G
#SBATCH -t 20:00:00
#SBATCH -n 1
#SBATCH -J MEDdef_hist
#SBATCH -o MEDdef_hist_%j.out
#SBATCH -e MEDdef_hist_%j.err

# If you need a specific Python module, uncomment and adjust:
# module load python/3.10

cd mpMED_1M
python ../hist_plotter_v1.py -t j -d mpMED_1M
