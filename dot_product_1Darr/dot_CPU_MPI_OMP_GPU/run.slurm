#!/bin/bash
#SBATCH -J dot_gpu_cap
#SBATCH -A ferrante
#SBATCH -p ckpt-all
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -t 00:15:00
#SBATCH -o dot_gpu_cap_%j.out

module purge
module load nvhpc/24.5
module load gcc/11.2.0

echo "nodes: $SLURM_JOB_NODELIST"
srun nvidia-smi -L
which nsys ncu || echo "nsys or ncu not on PATH"

mpif90 -O2 -Minfo=mp -mp=gpu -gpu=cc75,cc80,cc86,cc89,cc90 dot_hybrid_gpu_cap.f90 -o dot_hybrid_gpu_cap || exit 1
export OMP_TARGET_OFFLOAD=MANDATORY

echo "===== A: staged, no cap"
unset DOT_TEAMS DOT_THREADS
mpirun --bind-to none -x LD_LIBRARY_PATH -np 2 ./dot_hybrid_gpu_cap staged

echo "===== B: staged, capped at 26 teams x 128 threads = 3328 GPU threads"
export DOT_TEAMS=26 DOT_THREADS=128
mpirun --bind-to none -x LD_LIBRARY_PATH -x DOT_TEAMS -x DOT_THREADS -np 2 ./dot_hybrid_gpu_cap staged

echo "===== C: aware, capped"
export DOT_FORCE_AWARE=1
mpirun --bind-to none -x LD_LIBRARY_PATH -x DOT_TEAMS -x DOT_THREADS -x DOT_FORCE_AWARE -np 2 ./dot_hybrid_gpu_cap aware
