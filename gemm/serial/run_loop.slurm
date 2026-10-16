#!/bin/bash
#SBATCH -J gemm1_serial_loop
#SBATCH -A ferrante
#SBATCH -p compute-bigmem
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 00:20:00
#SBATCH -o gemm1_serial_loop_%j.out

set -eo pipefail

module purge
module load gcc/11.2.0

export OMP_NUM_THREADS=1

N=${1:-2048}
cd "$SLURM_SUBMIT_DIR"

echo "nodes=$SLURM_JOB_NUM_NODES ranks=1 N=$N implementation=Fortran_loops"

gfortran -O2 gemm_serial.f90 -o "gemm_serial_loop_${SLURM_JOB_ID}"

./"gemm_serial_loop_${SLURM_JOB_ID}" "$N"

# sbatch run_loop.slurm 2048
