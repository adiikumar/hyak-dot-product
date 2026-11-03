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

BUILD_DIR="$SLURM_SUBMIT_DIR/.job_build/$SLURM_JOB_ID"
mkdir -p "$BUILD_DIR"

EXE="$BUILD_DIR/gradient_CPU_MPI"
trap 'rm -rf -- "$BUILD_DIR"' EXIT

mpifort -O2 -J "$BUILD_DIR" \
  gradient_CPU_MPI.f90 -o "$EXE"

echo "N=$N nodes=$SLURM_JOB_NUM_NODES ranks=$SLURM_NTASKS threads=1"
echo "Executable=$EXE"

mpirun --bind-to none -np "$SLURM_NTASKS" \
  "$EXE" "$N"
