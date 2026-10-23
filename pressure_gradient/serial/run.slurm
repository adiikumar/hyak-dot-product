#!/bin/bash
#SBATCH -J pressure_gradient_serial
#SBATCH -A ferrante
#SBATCH -p compute-bigmem
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 00:05:00
#SBATCH -o pressure_gradient_serial%j.out

set -eo pipefail

module purge
module load gcc/11.2.0

cd "$SLURM_SUBMIT_DIR"

gfortran -O2 -fcheck=bounds pressure_gradient_serial.f90 -o pressure_gradient_serial

./pressure_gradient_serial
