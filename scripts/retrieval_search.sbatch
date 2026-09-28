#!/bin/bash
# Retrieval benchmark GPU jobs. Submit from the repo root.
#   feature cache (once):  sbatch --job-name=retr-precompute --array=0 scripts/retrieval_search.sbatch precompute
#   pilot:                 sbatch --array=0-1 scripts/retrieval_search.sbatch
#   full run (resumable):  sbatch --array=0-99%8 scripts/retrieval_search.sbatch
# Each array task searches query $SLURM_ARRAY_TASK_ID of datasets/retrieval/queries.csv
# against the full DB and skips itself if its hit list already exists.
#SBATCH --job-name=retr-search
#SBATCH --partition=gpu-h100-killable
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=/home/iscb/wolfson/hagairavid/LocAlign_retrieval_work/logs/%x-%A_%a.out
#SBATCH --error=/home/iscb/wolfson/hagairavid/LocAlign_retrieval_work/logs/%x-%A_%a.out

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"
PY=/home/iscb/wolfson/hagairavid/miniforge3/envs/localign_infer3/bin/python
cd "$REPO"
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID:-none} commit=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"

if [ "${1:-}" = "precompute" ]; then
    "$PY" scripts/retrieval_search.py --precompute --seed 42
else
    "$PY" scripts/retrieval_search.py --query_index "${SLURM_ARRAY_TASK_ID:?}" --seed 42
fi
echo "[$(date)] done"
