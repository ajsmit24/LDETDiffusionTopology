#!/bin/sh
#SBATCH -p et2024       
#SBATCH --mem=100G
#SBATCH -t 20:00:00              
#SBATCH -n 1
#SBATCH -J BIGdef
#SBATCH -o BIGdef_%j.out
#SBATCH -e BIGdef_%j.err


(cd mpBIG_1B; python ../revise_mPlot_v5.py j)
#(cd mpBIG_1B; mv rainbow_good.png default_rainbow.png)
