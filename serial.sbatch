#!/bin/bash
#SBATCH -J dot1_serial
#SBATCH -A ferrante
#SBATCH -p compute-bigmem
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:05:00
#SBATCH -o dot1_serial_%j.out
module purge
module load gcc/11.2.0
echo "nodes=$SLURM_JOB_NUM_NODES cpus_on_node=$SLURM_CPUS_ON_NODE host=$(hostname)"
gfortran -O2 dot_serial.f90 -o dot_serial
./dot_serial
