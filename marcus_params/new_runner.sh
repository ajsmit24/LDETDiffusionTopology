#!/bin/sh
#SBATCH -p et2024
#SBATCH --array=0-59           
#SBATCH --cpus-per-task=60
#SBATCH --mem=6G
#SBATCH -t 02:00:00              
#SBATCH -n 1
#SBATCH -J marcus_scan_array

echo "Task ID = $SLURM_ARRAY_TASK_ID"
echo "SLURM_CPUS_PER_TASK = $SLURM_CPUS_PER_TASK"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python marcus_mpi_v4.py \
    --num-chunks 60 \
    --chunk-index ${SLURM_ARRAY_TASK_ID} \
    --output mparams_chunk${SLURM_ARRAY_TASK_ID}.jsonl

echo "Chunk ${SLURM_ARRAY_TASK_ID} complete."
