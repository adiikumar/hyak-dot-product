#!/bin/bash
#SBATCH -J dot_aware2
#SBATCH -A ferrante
#SBATCH -p ckpt-all
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -c 4
#SBATCH -t 00:20:00
#SBATCH -o dot_aware2_%j.out
module purge
module load nvhpc/24.5
module load gcc/11.2.0
echo "nodes=$SLURM_JOB_NUM_NODES ntasks=$SLURM_NTASKS"
mpirun --bind-to none -x LD_LIBRARY_PATH -np $SLURM_NTASKS nvidia-smi -L
F="-O2 -mp=gpu -gpu=cc75,cc80,cc86,cc89,cc90"
mpif90 $F dot_aware_arr.f90 -o dot_aware_arr || exit 1
mpif90 $F dot_aware_ptr.f90 -o dot_aware_ptr || echo "ptr variant did not compile"
export OMP_TARGET_OFFLOAD=MANDATORY
R="mpirun --bind-to none -x LD_LIBRARY_PATH -x DOT_FORCE_AWARE -np $SLURM_NTASKS"
echo "===== staged"
$R ./dot_aware_arr staged
echo "===== aware with use_device_addr"
DOT_FORCE_AWARE=1 $R ./dot_aware_arr aware
echo "===== aware with use_device_ptr"
DOT_FORCE_AWARE=1 $R ./dot_aware_ptr aware
