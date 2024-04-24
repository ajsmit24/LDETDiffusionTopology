#!/bin/sh
#SBATCH -p et2024
#SBATCH -w et116
#SBATCH --mem-per-cpu=2000
#SBATCH -t 7000
#SBATCH -n 50
#SBATCH --output=%x.slurm
#SBATCH --error=%x.err

mpiexec -n 50 python -m mpi4py.futures pilot_cgValidation_mfpt_unidir.py
