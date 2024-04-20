#!/bin/sh
#SBATCH -p et3
#SBATCH --mem-per-cpu=1400
#SBATCH -t 7000
#SBATCH -n 20
#SBATCH --output=%x.slurm
#SBATCH --error=%x.err

mpiexec -n 20 python -m mpi4py.futures pilot_cgValidation_mft.py
