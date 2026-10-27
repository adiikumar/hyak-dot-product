#!/bin/bash
#SBATCH -J gemm3_mpi_omp
#SBATCH -A ferrante
#SBATCH -p compute-bigmem
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH -c 2
#SBATCH --hint=nomultithread
#SBATCH -t 00:05:00
#SBATCH -o gemm3_mpi_omp_%j.out

set -eo pipefail

module purge
module load gcc/13.2.0
module load ompi/4.1.6-2

# Two OpenMP CPU threads per MPI rank.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_DYNAMIC=FALSE

# Explicit CPU thread placement.
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_DISPLAY_AFFINITY=FALSE

# Prevent nested BLAS threading.
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1

N=${1:-2048}

cd "$SLURM_SUBMIT_DIR"

echo "nodes=$SLURM_JOB_NUM_NODES ranks=$SLURM_NTASKS threads=$OMP_NUM_THREADS N=$N"

# Ensure source exists.
if [ ! -s gemm_mpi_omp_blas.f90 ]; then
    echo "ERROR: gemm_mpi_omp_blas.f90 missing or empty"
    exit 1
fi

# Compile using OpenMPI, OpenMP and OpenBLAS.
mpif90 -O2 -fopenmp gemm_mpi_omp_blas.f90 \
  -o gemm_mpi_omp_blas \
  -L/sw/openblas/0.3.28/lib64 \
  -Wl,-rpath,/sw/openblas/0.3.28/lib64 \
  -lopenblas

# Execute once, with explicit MPI rank CPU affinity.
mpirun --bind-to none -np "$SLURM_NTASKS" \
  python3 -c '
import os, sys

local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
nthreads = int(os.environ["OMP_NUM_THREADS"])

allowed = sorted(os.sched_getaffinity(0))

if len(allowed) == nthreads:
    selected = allowed
else:
    start = local_rank*nthreads
    selected = allowed[start:start+nthreads]

if len(selected) != nthreads:
    sys.exit("ERROR: CPU assignment failed")

os.sched_setaffinity(0, selected)

os.execv(sys.argv[1], sys.argv[1:])
' ./gemm_mpi_omp_blas "$N"

# BASELINE COMMANDS: N=2048

# 1 node, 2 MPI ranks, 2 OpenMP threads/rank
# sbatch -N 1 -n 2 --ntasks-per-node=2 -c 2 run.slurm 2048

# 2 nodes, 4 MPI ranks, 2 OpenMP threads/rank
# sbatch -N 2 -n 4 --ntasks-per-node=2 -c 2 run.slurm 2048

# 4 nodes, 4 MPI ranks, 2 OpenMP threads/rank
# sbatch -N 4 -n 4 --ntasks-per-node=1 -c 2 run.slurm 2048
