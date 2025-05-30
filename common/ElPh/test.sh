#!/bin/sh
#SBATCH --partition=et2024,et3
#SBATCH --mem-per-cpu=1600
#SBATCH -t 7000
#SBATCH -n 200
#SBATCH --output=%x.slurm
#SBATCH --error=%x.err

mpiexec -n 200 python -m mpi4py.futures main.py
