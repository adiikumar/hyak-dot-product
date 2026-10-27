#!/bin/bash
#SBATCH -J pressure_mpi
#SBATCH -A ferrante
#SBATCH -p compute-bigmem
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --ntasks-per-node=2
#SBATCH -c 1
#SBATCH -t 00:05:00
#SBATCH -o pressure_mpi_%j.out

set -eo pipefail

module purge
module load gcc/13.2.0
module load ompi/4.1.6-2

cd "$SLURM_SUBMIT_DIR"

mpif90 -O2 -fcheck=bounds \
  pressure_gradient_mpi.f90 \
  -o pressure_gradient_mpi

echo "nodes=$SLURM_JOB_NUM_NODES ranks=$SLURM_NTASKS N=64"

mpirun --bind-to none ./pressure_gradient_mpi
