#!/bin/bash
#SBATCH -J gemm1_serial
#SBATCH -A ferrante
#SBATCH -p compute-bigmem
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 00:05:00
#SBATCH -o gemm1_serial_%j.out

set -eo pipefail

module purge
module load gcc/11.2.0
module load openblas/0.3.28

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1

N=${1:-2048}
cd "$SLURM_SUBMIT_DIR"

echo "nodes=$SLURM_JOB_NUM_NODES ranks=1 N=$N"

gfortran -O2 gemm_serial_blas.f90 -o gemm_serial_blas \
  -L/sw/openblas/0.3.28/lib64 \
  -Wl,-rpath,/sw/openblas/0.3.28/lib64 \
  -lopenblas

./gemm_serial_blas "$N"

# BASELINE: 1 node, 1 CPU core
# sbatch -N 1 -n 1 -c 1 run.slurm 2048
