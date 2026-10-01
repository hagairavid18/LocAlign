#!/bin/bash
# Validation-only apo/holo re-evaluation of the ligand-split LocAlign checkpoint
# (checkpoints/baseline-ligand-split/epoch=8-step=97191.ckpt), one subset per job.
#
# Usage (from the repository root; one job per subset):
#   mkdir -p logs/slurm
#   sbatch scripts/dispatch_apo_reval_ligckpt.sbatch apo_reval_ligand_holo_holo_subset_ligckpt
#   sbatch scripts/dispatch_apo_reval_ligckpt.sbatch apo_reval_ligand_apo_apo_ligckpt
#   sbatch scripts/dispatch_apo_reval_ligckpt.sbatch apo_reval_ligand_apo_holo_ligckpt
#
# Outputs: results/validation_results/<exp_name>/per_sample_results_0.csv
# Seed 41 matches scripts/dispatch_apo_reval.sbatch (and the trainer default).
#SBATCH --job-name=apo-reval-ligckpt
#SBATCH --partition=gpu-mol
#SBATCH --gres=gpu:1
#SBATCH --exclude=n-102
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/apo-reval-ligckpt-%j.out
#SBATCH --error=logs/slurm/apo-reval-ligckpt-%j.out

set -euo pipefail

CONFIG_NAME="${1:?usage: sbatch $0 <config name without .yaml>}"
SEED="${SEED:-41}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/slurm

ENV_BIN="/home/iscb/wolfson/hagairavid/miniforge3/envs/aligner_dl2/bin"
export PATH="${ENV_BIN}:${PATH}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} config=${CONFIG_NAME} seed=${SEED} cwd=$(pwd)"
nvidia-smi -L || true

# Fail fast if the allocated node has no usable CUDA device.
"${ENV_BIN}/python" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' \
    || { echo "CUDA not available on $(hostname); aborting"; exit 3; }

"${ENV_BIN}/python" aligner_dl/trainers/lightning_trainer.py \
    --config "aligner_dl/configs/${CONFIG_NAME}.yaml" \
    --log_dir "logs/${CONFIG_NAME}" \
    --seed "${SEED}"
