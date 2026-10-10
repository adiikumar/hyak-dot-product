#!/bin/bash
#SBATCH -J dot2_mpi
#SBATCH -A ferrante
#SBATCH -p compute-bigmem
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH -t 00:10:00
#SBATCH -o dot2_mpi_%j.out
module purge
module load gcc/11.2.0 ompi/4.1.6-2
echo "nodes=$SLURM_JOB_NUM_NODES cpus_on_node=$SLURM_CPUS_ON_NODE ntasks=$SLURM_NTASKS"
mpif90 -O2 dot_mpi.f90 -o dot_mpi
echo "--- rank placement (ranks per node)"
mpirun -np $SLURM_NTASKS hostname | sort | uniq -c
echo "--- MPI dot product, $SLURM_NTASKS ranks"
mpirun -np $SLURM_NTASKS ./dot_mpi
