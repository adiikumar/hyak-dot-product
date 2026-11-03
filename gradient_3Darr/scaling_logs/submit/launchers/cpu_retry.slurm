#!/bin/bash
#SBATCH --account=ferrante
#SBATCH --partition=ckpt-all
#SBATCH --time=00:08:00

set -euo pipefail

set +u
module purge
module load gcc/13.2.0
module load ligroup/openmpi/5.0.10
set -u

ROOT=/gscratch/ferrante/adityk7/test/gradient_3Darr
cd "$ROOT"

KIND="${1:?Specify mpi or omp}"
N=1024

NODES="$SLURM_JOB_NUM_NODES"
RANKS="$SLURM_NTASKS"
THREADS="$SLURM_CPUS_PER_TASK"
RPN=$((32 / THREADS))

if (( RANKS != NODES * RPN || RPN * THREADS != 32 )); then
    echo "ERROR: Incorrect CPU configuration"
    exit 2
fi

if [[ "$KIND" == mpi ]]; then
    SRC="$ROOT/gradient_CPU_MPI/gradient_CPU_MPI.f90"
    FLAGS=(-O2)
else
    SRC="$ROOT/gradient_CPU_MPI_OMP/gradient_CPU_MPI_OMP.f90"
    FLAGS=(-O2 -fopenmp)
fi

grep -q 'Norm_seconds' "$SRC" || {
    echo "ERROR: Expected instrumented Fortran source"
    exit 2
}

BUILD="$ROOT/.cpu_build/$SLURM_JOB_ID"
mkdir -p "$BUILD"
trap 'rm -rf -- "$BUILD"' EXIT

EXE="$BUILD/gradient"

mpifort "${FLAGS[@]}" -J "$BUILD" "$SRC" -o "$EXE"

export OMP_NUM_THREADS="$THREADS"
export OMP_DYNAMIC=FALSE
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DISPLAY_AFFINITY=TRUE

# Avoid the failing UCX DC transport; retain InfiniBand via RC.
export UCX_TLS=rc,sm,self

# Use slot mapping for 4-thread ranks.
if (( THREADS == 4 )); then
    MAP="slot:PE=4"
else
    MAP="ppr:${RPN}:node:PE=${THREADS}"
fi

echo "N=$N"
echo "Nodes=$NODES"
echo "MPI_ranks=$RANKS"
echo "Threads_per_rank=$THREADS"
echo "CPU_cores_per_node=32"
echo "UCX_TLS=$UCX_TLS"
echo "MPI_mapping=$MAP"
echo "Node=$(hostname)"

mpirun \
    --mca pml ucx \
    --mca btl '^uct' \
    -x UCX_TLS \
    --map-by "$MAP" \
    --bind-to core \
    --report-bindings \
    -np "$RANKS" \
    "$EXE" "$N"
