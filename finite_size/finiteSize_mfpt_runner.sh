#!/bin/sh
#SBATCH -p et3
#SBATCH --mem-per-cpu=1400
#SBATCH -t 7000
#SBATCH -n 15
#SBATCH --output=%x.slurm
#SBATCH --error=%x.err

mpiexec -n 15 python -m mpi4py.futures pilot_finiteSize_mfpt.py -n 3 -d 2 -c 120 -fdl 12 -j finitesize1
