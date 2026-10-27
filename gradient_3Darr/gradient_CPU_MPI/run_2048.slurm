#!/bin/bash
#SBATCH -J pressure_mpi
#SBATCH -A ferrante
#SBATCH -p ckpt-all
#SBATCH -N 16
#SBATCH -n 256
#SBATCH --ntasks-per-node=16
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o pressure_mpi_2048_%j.out

set -eo pipefail

module purge
module load gcc/13.2.0
module load ompi/4.1.6-2

N=${1:-2048}

cd "$SLURM_SUBMIT_DIR"

echo "nodes=$SLURM_JOB_NUM_NODES ranks=$SLURM_NTASKS N=$N"

mpif90 -O2 -fcheck=bounds \
  pressure_gradient_mpi_2048.f90 \
  -o pressure_gradient_mpi_2048

mpirun --bind-to none ./pressure_gradient_mpi_2048 "$N"
