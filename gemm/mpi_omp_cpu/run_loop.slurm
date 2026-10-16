#!/bin/bash
#SBATCH -J gemm3_mpi_omp_loop
#SBATCH -A ferrante
#SBATCH -p compute-bigmem
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH -c 2
#SBATCH --hint=nomultithread
#SBATCH -t 00:20:00
#SBATCH -o gemm3_mpi_omp_loop_%j.out

set -eo pipefail

module purge
module load gcc/13.2.0
module load ompi/4.1.6-2

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_DYNAMIC=FALSE
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

N=${1:-2048}
cd "$SLURM_SUBMIT_DIR"

echo "nodes=$SLURM_JOB_NUM_NODES ranks=$SLURM_NTASKS threads=$OMP_NUM_THREADS N=$N implementation=Fortran_loops"

exe="./gemm_mpi_omp_loop_${SLURM_JOB_ID}"

mpif90 -O2 -fopenmp gemm_mpi_omp.f90 -o "$exe"

mpirun --bind-to none -np "$SLURM_NTASKS" \
  python3 -c '
import os,sys

local_rank=int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
nthreads=int(os.environ["OMP_NUM_THREADS"])
allowed=sorted(os.sched_getaffinity(0))

selected=allowed if len(allowed)==nthreads else allowed[local_rank*nthreads:(local_rank+1)*nthreads]

if len(selected)!=nthreads:
    sys.exit("ERROR: CPU assignment failed")

os.sched_setaffinity(0,selected)
os.execv(sys.argv[1],sys.argv[1:])
' "$exe" "$N"

# 1 node / 2 ranks / 2 threads per rank
# sbatch -N 1 -n 2 --ntasks-per-node=2 -c 2 run_loop.slurm 2048

# 2 nodes / 4 ranks / 2 threads per rank
# sbatch -N 2 -n 4 --ntasks-per-node=2 -c 2 run_loop.slurm 2048

# 4 nodes / 4 ranks / 2 threads per rank
# sbatch -N 4 -n 4 --ntasks-per-node=1 -c 2 run_loop.slurm 2048
