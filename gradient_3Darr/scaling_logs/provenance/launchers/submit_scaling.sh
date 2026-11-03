#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N=2048

SERIAL="$ROOT/gradient_CPU_serial"
MPI="$ROOT/gradient_CPU_MPI"
OMP="$ROOT/gradient_CPU_MPI_OMP"

module purge
module load gcc/13.2.0
module load ligroup/openmpi/5.0.10

echo "Checking all three source files..."

gfortran -O2 -fsyntax-only \
    "$SERIAL/gradient_CPU_serial.f90"

mpifort -O2 -fsyntax-only \
    "$MPI/gradient_CPU_MPI.f90"

mpifort -O2 -fopenmp -fsyntax-only \
    "$OMP/gradient_CPU_MPI_OMP.f90"

echo "Syntax checks: PASS"

previous_job=""
count=0

submit_job() {
    local dir="$1"
    local name="$2"
    local partition="$3"
    local nodes="$4"
    local ranks="$5"
    local ppn="$6"
    local threads="$7"
    local memory="$8"
    local mesh="$9"

    local dependency=()

    if [[ -n "$previous_job" ]]; then
        dependency=(--dependency="afterok:$previous_job")
    fi

    local jobid

    jobid=$(
      cd "$dir"

      sbatch --parsable \
        "${dependency[@]}" \
        --account=ferrante \
        --partition="$partition" \
        --nodes="$nodes" \
        --ntasks="$ranks" \
        --ntasks-per-node="$ppn" \
        --cpus-per-task="$threads" \
        --mem="$memory" \
        --time=02:00:00 \
        --job-name="$name" \
        --output="${name}_%j.out" \
        run.slurm "$mesh"
    )

    previous_job="${jobid%%;*}"
    count=$((count+1))

    printf '%-24s N=%-5s Nodes=%-3s Ranks=%-4s Threads=%-2s Job=%s\n' \
      "$name" "$mesh" "$nodes" "$ranks" "$threads" "$previous_job"
}

echo
echo "========== SERIAL: MESH REFINEMENT =========="

for mesh in 128 256 512 1024 2048; do
    submit_job "$SERIAL" "SERIAL_N${mesh}" \
        compute-bigmem 1 1 1 1 300G "$mesh"
done

echo
echo "========== MPI: STRONG SCALING =========="

submit_job "$MPI" MPI_64  ckpt-all  8  64   8  1 48G "$N"
submit_job "$MPI" MPI_128 ckpt-all  8  128 16  1 48G "$N"
submit_job "$MPI" MPI_256 ckpt-all 16  256 16  1 48G "$N"
submit_job "$MPI" MPI_512 ckpt-all 16  512 32  1 48G "$N"

echo
echo "========== MPI + OPENMP: 1 THREAD =========="

submit_job "$OMP" OMP_64_T1  ckpt-all  8  64   8  1 48G "$N"
submit_job "$OMP" OMP_128_T1 ckpt-all  8  128 16  1 48G "$N"
submit_job "$OMP" OMP_256_T1 ckpt-all 16  256 16  1 48G "$N"
submit_job "$OMP" OMP_512_T1 ckpt-all 16  512 32  1 48G "$N"

echo
echo "========== MPI + OPENMP: 2 THREADS =========="

submit_job "$OMP" OMP_64_T2  ckpt-all  8  64   8  2 48G "$N"
submit_job "$OMP" OMP_128_T2 ckpt-all  8  128 16  2 48G "$N"
submit_job "$OMP" OMP_256_T2 ckpt-all 16  256 16  2 48G "$N"
submit_job "$OMP" OMP_512_T2 ckpt-all 16  512 32  2 48G "$N"

echo
echo "========== MPI + OPENMP: 4 THREADS =========="

submit_job "$OMP" OMP_64_T4  ckpt-all  8  64   8  4 48G "$N"
submit_job "$OMP" OMP_128_T4 ckpt-all  8  128 16  4 48G "$N"
submit_job "$OMP" OMP_256_T4 ckpt-all 16  256 16  4 48G "$N"
submit_job "$OMP" OMP_512_T4 ckpt-all 16  512 32  4 48G "$N"

echo
echo "=========================================="
echo "Submitted $count jobs"
echo "All jobs use afterok dependencies"
echo "Last job: $previous_job"
echo "=========================================="
