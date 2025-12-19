#!/bin/sh
#SBATCH -p et3
#SBATCH --mem-per-cpu=1250
#SBATCH -t 1800
#SBATCH -n 10



mpiexec -n 10 python -m mpi4py.futures marcus_scan_MPI.py
