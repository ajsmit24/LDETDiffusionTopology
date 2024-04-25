#!/bin/sh
#SBATCH -p {par}
#SBATCH --mem-per-cpu=2000
#SBATCH -t 7000
#SBATCH -n {cores}
#SBATCH --output=%x.slurm
#SBATCH --error=%x.err

mpiexec -n {cores} python -m mpi4py.futures ../pilot_cgValidation_mfpt_unidir.py -n {latnode} -d {maxdim} -c {cutoff} -j {jobname}
