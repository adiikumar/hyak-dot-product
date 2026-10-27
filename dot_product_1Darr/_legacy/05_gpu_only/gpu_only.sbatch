#!/bin/bash
#SBATCH -J dot_gpu_only
#SBATCH -A ferrante
#SBATCH -p ckpt-all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH -c 1
#SBATCH -t 00:10:00
#SBATCH -o dot_gpu_only_%j.out
module purge
module load nvhpc/24.5
module load gcc/11.2.0
echo "host=$(hostname)"
nvidia-smi -L
nvfortran -O2 -mp=gpu -gpu=cc75,cc80,cc86,cc89,cc90 dot_gpu_only.f90 -o dot_gpu_only || exit 1
export OMP_TARGET_OFFLOAD=MANDATORY
./dot_gpu_only
