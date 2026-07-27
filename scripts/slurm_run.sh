#!/bin/bash
# SLURM template for clover-cl (SPEC §11.1).
#
# Usage (single run OR matrix -- auto-detected from the config):
#   sbatch scripts/slurm_run.sh configs/<name>.yaml
#   sbatch scripts/slurm_run.sh configs/matrix_full.yaml
#
# Resumability: clover's Trainer detects existing checkpoints in the run
# directory and resumes automatically; run-matrix additionally skips cells
# whose status.json says "done". If this job is preempted or times out,
# resubmitting the *same* command just continues where it left off.
#
# Account/partition below are Snellius values (verify with `accinfo`);
# adjust for other sites.

#SBATCH --job-name=clover-run
#SBATCH --account=tesr125357
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: sbatch scripts/slurm_run.sh <config.yaml>" >&2
    exit 1
fi

CONFIG="$1"

# --- Site-specific module/venv activation hook -----------------------------
# Snellius example (adjust to the module versions you actually installed with):
# module load 2023
# module load Python/3.11.3-GCCcore-12.3.0
source .venv/bin/activate
# ----------------------------------------------------------------------------

echo "clover-cl: running ${CONFIG} on $(hostname), job ${SLURM_JOB_ID:-local}"

# A matrix config has a top-level `methods:` list; a single-run config does
# not. Batch jobs have no stdin, so run-matrix must get --confirm (its
# interactive prompt would die with EOFError otherwise).
if grep -qE '^methods:' "${CONFIG}"; then
    clover run-matrix "${CONFIG}" --confirm
else
    clover run "${CONFIG}"
fi
