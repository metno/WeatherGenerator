#!/bin/bash
set -euo pipefail

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
case "${1:-}" in
    --init)
        cli=$2
        shift 2
        exec "$cli" init "$@"
        ;;
    --array-worker)
        : "${SLURM_ARRAY_TASK_ID:?Must run as a Slurm array task}"
        exec "$2" write-year "$3" "$(($4 + SLURM_ARRAY_TASK_ID))"
        ;;
    --finalize)
        exec "$2" finalize "$3"
        ;;
esac

script=$(realpath -- "${BASH_SOURCE[0]}")
cli="$(dirname -- "$script")/.venv/bin/weathergen-subset"
if [[ ! -x "$cli" ]]; then
    echo "Install the standalone environment with uv sync --project $(dirname -- "$script")" >&2
    exit 2
fi

if [[ $# -lt 4 || ! "$3" =~ ^[1-9][0-9]{3}$ || ! "$4" =~ ^[1-9][0-9]{3}$ || "$3" -gt "$4" ]]; then
    echo "Usage: bash submit_subset.sh RAW OUTPUT FIRST_YEAR LAST_YEAR --south S --north N --west W --east E" >&2
    exit 2
fi

source=$(realpath -e -- "$1")
output=$(realpath -m -- "$2")
first=$3
last=$4
shift 4
if [[ -e "$output" ]]; then
    echo "Output already exists: $output" >&2
    exit 2
fi

logs="${output}.logs"
mkdir -p -- "$logs"
common=(--parsable --account=nonmifa1 --partition=gpil --qos=nf
        --nodes=1 --ntasks=1 --mem=8G)
init=$(sbatch "${common[@]}" --job-name=subset-init --cpus-per-task=1 --time=00:30:00 \
    --output="$logs/init-%j.log" -- "$script" --init "$cli" "$source" "$output" "$first" "$last" "$@")
echo "Initialize: $init"
array=$(sbatch "${common[@]}" --job-name=subset-write --cpus-per-task=2 --time=16:00:00 \
    --dependency="afterok:${init%%;*}" --kill-on-invalid-dep=yes \
    --array="0-$((last - first))%2" --output="$logs/write-%A_%a.log" \
    -- "$script" --array-worker "$cli" "$output" "$first")
echo "Write array: $array"
finalize=$(sbatch "${common[@]}" --job-name=subset-finalize --cpus-per-task=1 --time=00:30:00 \
    --dependency="afterok:${array%%;*}" --kill-on-invalid-dep=yes \
    --output="$logs/finalize-%j.log" -- "$script" --finalize "$cli" "$output")
echo "Finalize: $finalize"
echo "Output: $output"
echo "Logs: $logs"
