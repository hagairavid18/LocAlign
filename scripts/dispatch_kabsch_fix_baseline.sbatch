#!/bin/bash
#SBATCH --job-name=baseline-homology-kabsch-fix-h100
#SBATCH --partition=gpu-h100-killable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm/baseline-homology-kabsch-fix-h100-%j.out
#SBATCH --error=logs/slurm/baseline-homology-kabsch-fix-h100-%j.out

set -euo pipefail

REPO_ROOT="/home/iscb/wolfson/hagairavid/LocAlign/.claude/worktrees/synchronous-crafting-torvalds"
cd "$REPO_ROOT"
mkdir -p logs/slurm

source /home/iscb/wolfson/hagairavid/miniforge3/etc/profile.d/conda.sh
conda activate aligner_dl2

# The aligner_dl2 env has a site-packages .pth entry hardcoding the main
# repo's aligner_dl/ onto sys.path ahead of anything the script appends at
# runtime. Force this worktree's own aligner_dl/ to take priority so the
# code actually under test (not the main checkout) is what runs.
export PYTHONPATH="${REPO_ROOT}/aligner_dl${PYTHONPATH:+:${PYTHONPATH}}"

python aligner_dl/trainers/lightning_trainer.py \
    --config aligner_dl/configs/loc_align_kabsch_fix_baseline.yaml \
    --log_dir logs/baseline_homology_kabsch_fix_h100 \
    --seed 41
