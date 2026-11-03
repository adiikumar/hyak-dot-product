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

KIND="${1:?Expected mpi or omp}"
N=1024
NODES="$SLURM_JOB_NUM_NODES"
RANKS="$SLURM_NTASKS"
THREADS="$SLURM_CPUS_PER_TASK"
RPN=$((RANKS / NODES))

if (( THREADS < 1 || RANKS != NODES * (32 / THREADS) || RPN * THREADS != 32 )); then
  echo "ERROR: not 32 CPU cores/node" >&2
  exit 2
fi

case "$KIND" in
  mpi)
    [[ "$THREADS" == 1 ]] || exit 2
    SRC="$ROOT/gradient_CPU_MPI/gradient_CPU_MPI.f90"
    FLAGS=(-O2)
    ;;
  omp)
    SRC="$ROOT/gradient_CPU_MPI_OMP/gradient_CPU_MPI_OMP.f90"
    FLAGS=(-O2 -fopenmp)
    ;;
  *)
    echo "ERROR: invalid implementation" >&2
    exit 2
    ;;
esac

grep -q "Initialization_seconds" "$SRC" || {
  echo "ERROR: source lacks diagnostic timers" >&2
  exit 2
}

grep -q "Norm_seconds" "$SRC" || {
  echo "ERROR: source lacks norm timer" >&2
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

echo "N=$N nodes=$NODES ranks=$RANKS ranks_per_node=$RPN threads_per_rank=$THREADS"
echo "Node=$(hostname) CPU model=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
echo "MPI_mapping=ppr:${RPN}:node:PE=${THREADS}"

mpirun --map-by "ppr:${RPN}:node:PE=${THREADS}" \
  --bind-to core --report-bindings \
  -np "$RANKS" "$EXE" "$N"
