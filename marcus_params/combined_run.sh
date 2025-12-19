#!/bin/sh
#SBATCH -p et2024
#SBATCH --cpus-per-task=60      # use 60 cores on one node
#SBATCH --mem=6G
#SBATCH -t 02:00:00             # adjust as needed
#SBATCH -n 1
#SBATCH -J marcus_scan_full

echo "Running full Marcus scan on a single node"
echo "SLURM_CPUS_PER_TASK = $SLURM_CPUS_PER_TASK"

# if you use OpenMP anywhere (e.g. in BLAS), keep it consistent:
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Activate your environment here if needed
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env

python scan_and_res.py \
    --nprocs ${SLURM_CPUS_PER_TASK} 

echo "Full scan complete."
