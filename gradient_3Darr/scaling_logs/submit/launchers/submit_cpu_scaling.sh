#!/bin/bash
set -euo pipefail

ROOT=/gscratch/ferrante/adityk7/test/gradient_3Darr
LOGS="$ROOT/scaling_logs/cpu_fixed_v2"
LEDGER="$LOGS/submissions.csv"

cd "$ROOT"
mkdir -p "$LOGS"

bash -n "$ROOT/run_cpu_scaling.slurm"

for src in \
  "$ROOT/gradient_CPU_MPI/gradient_CPU_MPI.f90" \
  "$ROOT/gradient_CPU_MPI_OMP/gradient_CPU_MPI_OMP.f90"
do
  [[ -f "$src" ]] || { echo "Missing $src" >&2; exit 2; }
  grep -q 'Initialization_seconds' "$src" || {
    echo "Unfixed source: $src" >&2
    exit 2
  }
done

if [[ ! -f "$LEDGER" ]]; then
  printf 'job_name,nodes,mpi_ranks,threads_per_rank,job_id\n' > "$LEDGER"
fi

for nodes in 2 4 8; do
  for variant in mpi omp_t1 omp_t2 omp_t4; do

    case "$variant" in
      mpi) kind=mpi; threads=1 ;;
      omp_t1) kind=omp; threads=1 ;;
      omp_t2) kind=omp; threads=2 ;;
      omp_t4) kind=omp; threads=4 ;;
    esac

    ppn=$((32 / threads))
    ranks=$((nodes * ppn))
    name="${variant}_nodes${nodes}"

    if grep -q "^${name}," "$LEDGER"; then
      echo "SKIP already submitted: $name"
      continue
    fi

    returned=$(sbatch --parsable \
      --account=ferrante \
      --partition=ckpt-all \
      --nodes="$nodes" \
      --ntasks="$ranks" \
      --ntasks-per-node="$ppn" \
      --cpus-per-task="$threads" \
      --hint=nomultithread \
      --mem=96G \
      --time=00:08:00 \
      --job-name="$name" \
      --output="$LOGS/${name}_%j.out" \
      "$ROOT/run_cpu_scaling.slurm" "$kind")

    id="${returned%%;*}"

    printf '%s,%s,%s,%s,%s\n' \
      "$name" "$nodes" "$ranks" "$threads" "$id" \
      >> "$LEDGER"

    echo "SUBMITTED $name -> $id ($nodes nodes, $ranks ranks, ${threads}T/rank)"
  done
done

echo "New outputs: $LOGS"
echo "Registry:    $LEDGER"
