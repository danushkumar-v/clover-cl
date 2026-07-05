#!/bin/bash
# SLURM template for a single clover-cl run (SPEC §11.1).
#
# Usage:
#   sbatch scripts/slurm_run.sh configs/<name>.yaml
#
# Resumability: clover's Trainer detects existing checkpoints in the run
# directory and resumes automatically (clover/training/trainer.py) -- if
# this job is preempted or times out, resubmitting the *same* command with
# the same config just continues from the last completed experience. No
# separate --resume flag is needed.
#
# The #SBATCH lines below are placeholders. Fill in your site's account,
# partition, and time budget before submitting; see your cluster's
# documentation for the correct values (this is not runnable as-is).

#SBATCH --job-name=clover-run
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: sbatch scripts/slurm_run.sh <config.yaml>" >&2
    exit 1
fi

CONFIG="$1"

# --- Site-specific module/venv activation hook -----------------------------
# module load python/3.11 cuda/12.1        # example; adjust for your site
# source .venv/bin/activate                # or: conda activate clover
# ----------------------------------------------------------------------------

echo "clover-cl: running ${CONFIG} on $(hostname), job ${SLURM_JOB_ID:-local}"
clover run "${CONFIG}"
