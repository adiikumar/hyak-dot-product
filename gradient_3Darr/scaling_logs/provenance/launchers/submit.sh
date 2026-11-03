#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MPI="$ROOT/gradient_CPU_MPI"
OMP="$ROOT/gradient_CPU_MPI_OMP"
GPU="$ROOT/gradient_CPU_MPI_OMP_GPU"
SERIAL="$ROOT/gradient_CPU_serial"
LOGS="$ROOT/scaling_logs"

PARTITION=ckpt-all
ACCOUNT=ferrante
WALLTIME=00:05:00

MODE=cpu
DRY=0
GPU_CONFIRMED=0

for arg in "$@"; do
    case "$arg" in
        cpu|gpu|all) MODE="$arg" ;;
        --dry-run) DRY=1 ;;
        --confirm-gpu-args) GPU_CONFIRMED=1 ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

if [[ "$MODE" != cpu && "$DRY" -eq 0 && "$GPU_CONFIRMED" -ne 1 ]]; then
    echo "GPU blocked: verify GPU run.slurm arguments first." >&2
    exit 2
fi

if (( ! DRY )); then
    if [[ "$MODE" != gpu ]]; then
        set +u
        module purge
        module load gcc/13.2.0
        module load ligroup/openmpi/5.0.10
        set -u

        gfortran -O2 -fsyntax-only \
            "$SERIAL/gradient_CPU_serial.f90"
        mpifort -O2 -fsyntax-only \
            "$MPI/gradient_CPU_MPI.f90"
        mpifort -O2 -fopenmp -fsyntax-only \
            "$OMP/gradient_CPU_MPI_OMP.f90"

        echo "CPU syntax checks PASS"
    fi

    if [[ "$MODE" != cpu && ! -f "$GPU/run.slurm" ]]; then
        echo "GPU run.slurm not found" >&2
        exit 1
    fi

    mkdir -p "$LOGS"
fi

cpu_mem() {
    case "$1:$2" in
        256:*) echo 16G ;;
        512:*) echo 32G ;;
        1024:1) echo 160G ;;
        1024:2) echo 96G ;;
        1024:4) echo 64G ;;
        1024:8) echo 48G ;;
        *) echo 48G ;;
    esac
}

COUNT=0

submit_cpu() {
    local kind="$1"
    local nodes="$2"
    local threads="$3"
    local n="$4"

    local ppn=$((32 / threads))
    local ranks=$((nodes * ppn))
    local dir name jobid

    if [[ "$kind" == MPI ]]; then
        dir="$MPI"
        name="MPI_N${n}_nodes${nodes}"
    else
        dir="$OMP"
        name="OMP_T${threads}_N${n}_nodes${nodes}"
    fi

    printf '%-26s N=%-5s nodes=%-2s ranks=%-3s ppn=%-2s threads=%s' \
        "$name" "$n" "$nodes" "$ranks" "$ppn" "$threads"

    if (( DRY )); then
        echo " [DRY RUN]"
    else
        jobid=$(
            cd "$dir"
            sbatch --parsable \
                -A "$ACCOUNT" \
                -p "$PARTITION" \
                -N "$nodes" \
                -n "$ranks" \
                --ntasks-per-node="$ppn" \
                -c "$threads" \
                --hint=nomultithread \
                --mem="$(cpu_mem "$n" "$nodes")" \
                -t "$WALLTIME" \
                -J "$name" \
                -o "$LOGS/${name}_%j.out" \
                --export="ALL,OMP_NUM_THREADS=$threads,OMP_PLACES=cores,OMP_PROC_BIND=close" \
                run.slurm "$n"
        )
        echo " Job=${jobid%%;*}"
    fi

    COUNT=$((COUNT + 1))
}

submit_gpu() {
    local nodes="$1"
    local ng="$2"
    local n="$3"
    local nr="$1"
    local name="GPU_G${ng}_N${n}_nodes${nodes}"
    local jobid

    printf '%-26s N=%-5s nodes=%-2s MPI_ranks=%-2s GPUs/node=%s' \
        "$name" "$n" "$nodes" "$nr" "$ng"

    if (( DRY )); then
        echo " [DRY RUN]"
    else
        # Requires run.slurm $1=N, $2=Ng.
        # One MPI rank per node.
        jobid=$(
            cd "$GPU"
            sbatch --parsable \
                -A "$ACCOUNT" \
                -p "$PARTITION" \
                -N "$nodes" \
                -n "$nr" \
                --ntasks-per-node=1 \
                -c 1 \
                --gpus-per-node="a40:$ng" \
                --mem=96G \
                -t "$WALLTIME" \
                -J "$name" \
                -o "$LOGS/${name}_%j.out" \
                --export="ALL,NR=$nr,NG=$ng,OMP_NUM_THREADS=1" \
                run.slurm "$n" "$ng"
        )
        echo " Job=${jobid%%;*}"
    fi

    COUNT=$((COUNT + 1))
}

if [[ "$MODE" == cpu || "$MODE" == all ]]; then

    echo "=== FIG 2: CPU STRONG SCALING, N=1024 ==="

    for nodes in 1 2 4 8; do
        submit_cpu MPI "$nodes" 1 1024

        for threads in 1 2 4; do
            submit_cpu OMP "$nodes" "$threads" 1024
        done
    done

    echo "=== FIG 1: ADDITIONAL CPU CONVERGENCE ==="

    for n in 256 512; do
        submit_cpu MPI 1 1 "$n"
        submit_cpu MPI 2 1 "$n"
        submit_cpu OMP 2 2 "$n"
    done
fi

if [[ "$MODE" == gpu || "$MODE" == all ]]; then

    echo "=== FIG 2: MISSING GPU CONFIGURATIONS ==="

    submit_gpu 1 4 1024
    submit_gpu 2 1 1024
    submit_gpu 4 1 1024
    submit_gpu 4 4 1024
    submit_gpu 8 4 1024

    echo "=== FIG 1: ADDITIONAL GPU CONVERGENCE ==="

    submit_gpu 2 2 256
    submit_gpu 2 2 512
fi

echo "Total $COUNT independent jobs $([[ "$DRY" -eq 1 ]] && echo '(preview only)' || true)"
echo "No afterok dependencies; each job requests five minutes."
