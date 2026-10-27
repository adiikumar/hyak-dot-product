#!/bin/bash
#SBATCH -J gemm2_mpi
#SBATCH -A ferrante
#SBATCH -p compute-bigmem
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH -c 1
#SBATCH --hint=nomultithread
#SBATCH -t 00:05:00
#SBATCH -o gemm2_mpi_%j.out

set -eo pipefail

module purge
module load gcc/13.2.0
module load ompi/4.1.6-2

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1

N=${1:-2048}
cd "$SLURM_SUBMIT_DIR"

echo "nodes=$SLURM_JOB_NUM_NODES ranks=$SLURM_NTASKS N=$N"

mpif90 -O2 gemm_mpi_blas.f90 -o gemm_mpi_blas \
  -L/sw/openblas/0.3.28/lib64 \
  -Wl,-rpath,/sw/openblas/0.3.28/lib64 \
  -lopenblas

mpirun --bind-to none -np "$SLURM_NTASKS" \
  python3 -c '
import os, sys
local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
allowed = sorted(os.sched_getaffinity(0))
selected = allowed if len(allowed) == 1 else allowed[local_rank:local_rank+1]
if len(selected) != 1:
    sys.exit("ERROR: CPU assignment failed")
os.sched_setaffinity(0, selected)
os.execv(sys.argv[1], sys.argv[1:])
' ./gemm_mpi_blas "$N"

# BASELINE COMMANDS
# 1 node, 2 MPI ranks
# sbatch -N 1 -n 2 --ntasks-per-node=2 -c 1 run.slurm 2048

# 2 nodes, 4 MPI ranks
# sbatch -N 2 -n 4 --ntasks-per-node=2 -c 1 run.slurm 2048

# 4 nodes, 4 MPI ranks
# sbatch -N 4 -n 4 --ntasks-per-node=1 -c 1 run.slurm 2048
