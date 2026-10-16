#!/bin/bash
#SBATCH -J gemm4_mpi_gpu
#SBATCH -A ferrante
#SBATCH -p ckpt-all
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH -c 1
#SBATCH -t 00:15:00
#SBATCH -o gemm4_mpi_gpu_%j.out

set -eo pipefail

module purge
module load nvhpc/24.5
module load gcc/11.2.0

N=${1:-2048}
export GPUS_PER_RANK=${2:-1}

export OMP_NUM_THREADS="$GPUS_PER_RANK"
export OMP_DYNAMIC=FALSE
export OMP_TARGET_OFFLOAD=MANDATORY

cd "$SLURM_SUBMIT_DIR"

echo "nodes=$SLURM_JOB_NUM_NODES ranks=$SLURM_NTASKS GPUs/rank=$GPUS_PER_RANK N=$N"

if [ "$SLURM_CPUS_PER_TASK" -lt "$GPUS_PER_RANK" ]; then
    echo "ERROR: insufficient host CPUs per rank"
    exit 1
fi

mpif90 -O2 -mp=gpu -gpu=cc75,cc80,cc86,cc89,cc90 \
  gemm_mpi_gpu_omp.f90 -o gemm_mpi_gpu_omp

mpirun --bind-to none \
  -x LD_LIBRARY_PATH \
  -x GPUS_PER_RANK \
  -x OMP_NUM_THREADS \
  -x OMP_TARGET_OFFLOAD \
  -np "$SLURM_NTASKS" \
  ./gemm_mpi_gpu_omp "$N"

# BASELINE COMMANDS
# 1 node, 2 MPI ranks, 2 GPUs
# sbatch -N 1 -n 2 --ntasks-per-node=2 --gpus-per-node=2 -c 1 run.slurm 2048 1

# 2 nodes, 4 MPI ranks, 4 GPUs
# sbatch -N 2 -n 4 --ntasks-per-node=2 --gpus-per-node=2 -c 1 run.slurm 2048 1

# 4 nodes, 4 MPI ranks, 4 GPUs
# sbatch -N 4 -n 4 --ntasks-per-node=1 --gpus-per-node=1 -c 1 run.slurm 2048 1
