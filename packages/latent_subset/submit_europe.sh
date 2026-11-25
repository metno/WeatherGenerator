#!/bin/bash
#SBATCH --account=nonmifa1
#SBATCH --partition=gpil
#SBATCH --qos=nf
#SBATCH --job-name=subset-europe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=16:00:00
#SBATCH --array=0-8%2
#SBATCH --output=subset-europe-%A_%a.out
#SBATCH --error=subset-europe-%A_%a.err

set -euo pipefail

: "${SLURM_SUBMIT_DIR:?Submit from the repository root using sbatch}"
: "${SLURM_ARRAY_TASK_ID:?One array task is required per year}"
if [[ $# -ne 2 ]]; then
    echo "Usage: sbatch submit_europe.sh RAW_ARCHIVE OUTPUT_ROOT" >&2
    exit 2
fi

year=$((2016 + SLURM_ARRAY_TASK_ID))
source="$1/$year"
output="$2/$year"
cli="$SLURM_SUBMIT_DIR/packages/latent_subset/.venv/bin/weathergen-subset"
if [[ ! -x "$cli" ]]; then
    echo "Standalone package environment not found: run uv sync --project packages/latent_subset" >&2
    exit 2
fi
if [[ ! -d "$source" ]]; then
    echo "Source year directory does not exist: $source" >&2
    exit 2
fi
if [[ -e "$output" ]]; then
    echo "Output already exists: $output (use a fresh output root to retry)" >&2
    exit 2
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Subsetting $source to $output on $(hostname)"
srun --ntasks=1 "$cli" "$source" "$output" \
    --south 34 --north 72 --west -12 --east 60
