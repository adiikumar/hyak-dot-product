#!/bin/bash
#SBATCH --account=ferrante
#SBATCH --partition=ckpt-all
#SBATCH --time=00:05:00

set -euo pipefail

module purge
module load gcc/13.2.0
module load nvhpc/24.5

cd "$SLURM_SUBMIT_DIR"

N="${1:-1024}"
GPUS_PER_RANK="${2:-1}"

# Store NVIDIA compiler configuration in lab scratch.
LAB_CACHE="$SLURM_SUBMIT_DIR/.nvhpc_job_cache/$SLURM_JOB_ID"
COMPILER_HOME="$LAB_CACHE/home"
COMPILER_TMP="$LAB_CACHE/tmp"

mkdir -p "$COMPILER_HOME/.config" \
         "$COMPILER_HOME/.cache" \
         "$COMPILER_TMP"

EXE="./gradient_CPU_MPI_OMP_GPU_${SLURM_JOB_ID}"
trap 'rm -f "$EXE"' EXIT

ATOMIC_LIB="$(gcc -print-file-name=libatomic.so.1)"

if [[ ! -f "$ATOMIC_LIB" ]]; then
    echo "ERROR: libatomic.so.1 not found"
    exit 2
fi

export LD_LIBRARY_PATH="$(dirname "$ATOMIC_LIB")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "N=$N"
echo "Nodes=${SLURM_JOB_NUM_NODES}"
echo "MPI ranks=${SLURM_NTASKS}"
echo "GPUs per rank=$GPUS_PER_RANK"
echo "Compiler cache=$LAB_CACHE"

echo "Compiling..."

# Redirect compiler-generated HOME, XDG, and temporary files
# into the lab directory. Preserve normal HOME at runtime.
(
    export HOME="$COMPILER_HOME"
    export XDG_CONFIG_HOME="$COMPILER_HOME/.config"
    export XDG_CACHE_HOME="$COMPILER_HOME/.cache"
    export TMPDIR="$COMPILER_TMP"

    mpifort -O2 -mp=gpu -gpu=cc86 -Minfo=mp \
      gradient_CPU_MPI_OMP_GPU.f90 -o "$EXE"
)

if ldd "$EXE" | grep -q 'not found'; then
    echo "ERROR: Missing runtime library"
    ldd "$EXE"
    exit 3
fi

export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OMP_TARGET_OFFLOAD=MANDATORY

echo "GPUs allocated:"
nvidia-smi --query-gpu=index,uuid,name \
  --format=csv,noheader

echo "Starting computation..."

mpirun --bind-to none -x LD_LIBRARY_PATH \
  "$EXE" "$N" "$GPUS_PER_RANK"

echo "Completed."
