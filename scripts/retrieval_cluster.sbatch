#!/bin/bash
# Retrieval benchmark step 0 (CPU only): seed the shared feature cache from the
# earlier full-DB search (hard links, no extra disk), link one structure per DB
# chain, and cluster the chains with Foldseek at TM-score >= 0.6.
#SBATCH --job-name=retr-cluster
#SBATCH --partition=cpu-killable
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/iscb/wolfson/hagairavid/LocAlign_retrieval_work/logs/cluster-%j.out
#SBATCH --error=/home/iscb/wolfson/hagairavid/LocAlign_retrieval_work/logs/cluster-%j.out

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"
WORK=/home/iscb/wolfson/hagairavid/LocAlign_retrieval_work
OLD_CACHE=/home/iscb/wolfson/hagairavid/LocAlign/inference_1iep_sti/.cache
PY=/home/iscb/wolfson/hagairavid/miniforge3/envs/localign_infer3/bin/python
FOLDSEEK=/home/iscb/wolfson/hagairavid/miniforge3/envs/foldseek/bin/foldseek
cd "$REPO"

mkdir -p "$WORK/search/.cache"
if [ ! -f "$WORK/search/.cache/.seeded" ]; then
    echo "[$(date)] hard-linking cache from $OLD_CACHE"
    cp -a -l -n "$OLD_CACHE/." "$WORK/search/.cache/"
    touch "$WORK/search/.cache/.seeded"
fi

echo "[$(date)] linking chains"
"$PY" scripts/retrieval_prepare.py link-chains

mkdir -p "$WORK/foldseek"
CMD="$FOLDSEEK easy-cluster $WORK/chains $WORK/foldseek/clu_tm06 $WORK/foldseek/tmp --alignment-type 1 --tmscore-threshold 0.6 --threads ${SLURM_CPUS_PER_TASK:-32}"
echo "[$(date)] $CMD"
$CMD
"$FOLDSEEK" version

"$PY" scripts/retrieval_prepare.py clusters-csv
echo "[$(date)] done"
