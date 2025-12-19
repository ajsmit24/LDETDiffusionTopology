#!/bin/sh
#SBATCH -p et2
#SBATCH --mem-per-cpu=1250
#SBATCH -t 2400
#SBATCH -n 20


#Rnn lambda coupling
mpiexec -n 10 python -m mpi4py.futures marcus_scan_MPI.py 60 120 100
