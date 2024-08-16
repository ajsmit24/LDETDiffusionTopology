#!/bin/sh
#SBATCH -p et2024
#SBATCH --mem=320000
#SBATCH -t 7000
#SBATCH -n 200
#SBATCH --output=%x.slurm
#SBATCH --error=%x.err

mpiexec -n 200 python -m mpi4py.futures main.py
