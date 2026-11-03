#!/bin/bash
set -euo pipefail

ROOT=/gscratch/ferrante/adityk7/test/gradient_3Darr
LOGS="$ROOT/scaling_logs/cpu_fixed_v2/retries"
LEDGER="$LOGS/submissions.csv"

mkdir -p "$LOGS"
cd "$ROOT"

bash -n cpu_retry.slurm

if [[ ! -f "$LEDGER" ]]; then
    echo "name,old_job_id,new_job_id,nodes,ranks,threads" > "$LEDGER"
fi

submit_one() {
    local name="$1"
    local old="$2"
    local nodes="$3"
    local threads="$4"
    local kind="$5"

    local ppn=$((32 / threads))
    local ranks=$((nodes * ppn))
    local result id

    if grep -q "^${name}," "$LEDGER"; then
        echo "SKIP: $name already resubmitted"
        return
    fi

    result=$(sbatch --parsable \
        -A ferrante \
        -p ckpt-all \
        -N "$nodes" \
        -n "$ranks" \
        --ntasks-per-node="$ppn" \
        --cpus-per-task="$threads" \
        --hint=nomultithread \
        --mem=96G \
        -t 00:08:00 \
        -J "$name" \
        -o "$LOGS/${name}_%j.out" \
        cpu_retry.slurm "$kind")

    id="${result%%;*}"

    echo "$name,$old,$id,$nodes,$ranks,$threads" >> "$LEDGER"
    echo "$name: OLD $old -> NEW $id"
}

submit_one mpi_nodes2    42732317 2 1 mpi
submit_one mpi_nodes8    42732325 8 1 mpi
submit_one omp_t1_nodes8 42732326 8 1 omp
submit_one omp_t4_nodes4 42732324 4 4 omp
submit_one omp_t4_nodes8 42732328 8 4 omp

echo "Retry logs: $LOGS"
