#!/bin/bash
#SBATCH --job-name=kabsch-fix-smoketest
#SBATCH --partition=gpu-mol
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/kabsch-fix-smoketest-%j.out
#SBATCH --error=logs/slurm/kabsch-fix-smoketest-%j.out

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
    --config aligner_dl/configs/loc_align_kabsch_fix_smoketest.yaml \
    --log_dir logs/kabsch_fix_smoketest \
    --seed 41
