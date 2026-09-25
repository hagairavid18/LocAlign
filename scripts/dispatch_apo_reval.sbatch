#!/bin/bash
#SBATCH --job-name=apo-reval
#SBATCH --partition=gpu-mol
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/apo-reval-%j.out
#SBATCH --error=logs/slurm/apo-reval-%j.out

set -euo pipefail

REPO_ROOT="/home/iscb/wolfson/hagairavid/LocAlign"
cd "$REPO_ROOT"
mkdir -p logs/slurm

ENV_BIN="/home/iscb/wolfson/hagairavid/miniforge3/envs/aligner_dl2/bin"
export PATH="${ENV_BIN}:${PATH}"

CONFIGS=(
    apo_reval_homology_holo_holo_subset
    apo_reval_homology_apo_apo
    apo_reval_homology_apo_holo
    apo_reval_ligand_holo_holo_subset
    apo_reval_ligand_apo_apo
    apo_reval_ligand_apo_holo
)

for name in "${CONFIGS[@]}"; do
    echo "=== running ${name} ==="
    "${ENV_BIN}/python" aligner_dl/trainers/lightning_trainer.py \
        --config "aligner_dl/configs/${name}.yaml" \
        --log_dir "logs/${name}" \
        --seed 41
done
