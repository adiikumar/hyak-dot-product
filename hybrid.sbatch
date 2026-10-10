#!/bin/bash
#SBATCH -J dot3_hybrid
#SBATCH -A ferrante
#SBATCH -p compute-bigmem
#SBATCH -N 2
#SBATCH --ntasks-per-node=2
#SBATCH -c 2
#SBATCH -t 00:10:00
#SBATCH -o dot3_hybrid_%j.out
module purge
module load gcc/11.2.0 ompi/4.1.6-2
echo "nodes=$SLURM_JOB_NUM_NODES ntasks=$SLURM_NTASKS cpus_per_task=$SLURM_CPUS_PER_TASK"
gfortran -O2 -fopenmp dot_omp.f90 -o dot_omp
mpif90 -O2 -fopenmp dot_hybrid.f90 -o dot_hybrid
echo "===== 3a: OpenMP only, one node"
for T in 1 2; do ./dot_omp $T; done
echo "===== 3b: MPI + OpenMP, $SLURM_NTASKS ranks"
for T in 1 2; do
  export OMP_NUM_THREADS=$T
  echo "--- OMP_NUM_THREADS=$T"
  mpirun -np $SLURM_NTASKS --bind-to none ./dot_hybrid
done
