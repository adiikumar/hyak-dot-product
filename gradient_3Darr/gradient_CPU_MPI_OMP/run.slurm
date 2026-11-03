#!/bin/bash
#SBATCH --account=ferrante
#SBATCH --partition=ckpt-all
#SBATCH --time=00:05:00

set -euo pipefail

set +u
module purge
module load gcc/13.2.0
module load ligroup/openmpi/5.0.10
set -u

cd "$SLURM_SUBMIT_DIR"

N="${1:-1024}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC=FALSE

BUILD_DIR="$SLURM_SUBMIT_DIR/.job_build/$SLURM_JOB_ID"
mkdir -p "$BUILD_DIR"

EXE="$BUILD_DIR/gradient_CPU_MPI_OMP"
trap 'rm -rf -- "$BUILD_DIR"' EXIT

mpifort -O2 -fopenmp -J "$BUILD_DIR" \
  gradient_CPU_MPI_OMP.f90 -o "$EXE"

echo "N=$N nodes=$SLURM_JOB_NUM_NODES ranks=$SLURM_NTASKS threads=$OMP_NUM_THREADS"
echo "Executable=$EXE"

mpirun --bind-to none -np "$SLURM_NTASKS" \
  "$EXE" "$N"
