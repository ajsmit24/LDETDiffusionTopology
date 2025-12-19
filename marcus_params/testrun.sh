#!/bin/sh
#SBATCH -p et3
#SBATCH --mem=2G
#SBATCH -t 600
#SBATCH -n 5



mpiexec -n 5 python -m mpi4py.futures marcus_scan_MPI.py 10
