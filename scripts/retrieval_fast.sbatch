#!/bin/bash
# Retrieval benchmark, fast path (packed features, several queries per task).
# Submit from the repo root, e.g. 20 tasks x 5 queries:
#   sbatch --array=0-19%6 --exclude=n-102 scripts/retrieval_fast.sbatch 5
# (this cluster rejects multi-partition requests; override with --partition=gpu-h200-killable)
# Task g handles queries [g*Q, (g+1)*Q). Finished queries are skipped and per-shard
# partial scores are reused, so a preempted/requeued task resumes where it stopped.
#SBATCH --job-name=retr-fast
#SBATCH --partition=gpu-h100-killable
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --output=/home/iscb/wolfson/hagairavid/LocAlign_retrieval_work/logs/%x-%A_%a.out
#SBATCH --error=/home/iscb/wolfson/hagairavid/LocAlign_retrieval_work/logs/%x-%A_%a.out

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"
PY=/home/iscb/wolfson/hagairavid/miniforge3/envs/localign_infer3/bin/python
GROUP_SIZE="${1:-5}"
cd "$REPO"
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID:-none} restart=${SLURM_RESTART_COUNT:-0} commit=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"
nvidia-smi -L || true
# Node-local storage used for the staged shard copy (scripts/retrieval_stage.py).
echo "TMPDIR=${TMPDIR:-unset}"; df -h /tmp /dev/shm 2>/dev/null || true
ls -ld /tmp/localign_retrieval_${USER}/packed_* 2>/dev/null || echo "no staged shard copy on this node yet"
# Fail fast if CUDA is not usable (inference would silently fall back to CPU).
"$PY" -c "import torch,sys; ok=torch.cuda.is_available(); print('CUDA available:', ok, 'torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 3)"

"$PY" scripts/retrieval_fast.py --group "${SLURM_ARRAY_TASK_ID:?}" --group_size "$GROUP_SIZE" --seed 42 ${RETR_EXTRA_ARGS:-}
echo "[$(date)] done"
