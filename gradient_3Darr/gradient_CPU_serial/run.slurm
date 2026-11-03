#!/bin/bash
#SBATCH --account=ferrante
#SBATCH --partition=compute-bigmem
#SBATCH --time=02:00:00

set -euo pipefail

module purge
module load gcc/13.2.0

cd "$SLURM_SUBMIT_DIR"

N="${1:-2048}"

gfortran -O2 gradient_CPU_serial.f90 -o gradient_CPU_serial

echo "N=$N nodes=1 ranks=1 threads=1"
./gradient_CPU_serial "$N"
