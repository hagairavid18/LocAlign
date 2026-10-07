"""A posteriori split of each released train.csv into train and val.

The released checkpoints were trained on the whole of train.csv. This split was made
afterwards, so that train.csv can be described as train + val and future work can tune on
val without touching test.csv. It played no role in training the released checkpoints.

Writes <split dir>/train_val_assignment.csv with one row per row of train.csv, in the same
order: `row` (0-based row of train.csv) and `split`:
  homology_25_10 (cluster-based): a fraction POSTHOC_VAL_CLUSTER_FRACTION of the MMseqs2
      clusters of train.csv (tar_cluster/src_cluster, 50% identity, as in split_dataset.py) is
      drawn with seed POSTHOC_VAL_SEED. A pair is 'val' if both chains are in those clusters,
      'train' if neither is, and 'straddling' otherwise. Straddling pairs share a cluster
      with val, so they belong to neither side; they were in the training data of the
      released checkpoint.
  ligand_25_10 (ligand-based): a fraction POSTHOC_VAL_LIGAND_FRACTION of the ligand_ids is
      drawn with the same seed; their pairs are 'val', all others 'train'.

Usage
  python aligner_dl/datasets/utils/posthoc_val_split.py
"""
import argparse
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.constants import (  # noqa: E402
    DATASETS_DIR,
    POSTHOC_VAL_ASSIGNMENT_CSV,
    POSTHOC_VAL_CLUSTER_FRACTION,
    POSTHOC_VAL_LIGAND_FRACTION,
    POSTHOC_VAL_SEED,
    POSTHOC_VAL_SPLITS,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv_dir', default=os.path.join(DATASETS_DIR, 'csv_files'))
    args = ap.parse_args()
    for split, mode in POSTHOC_VAL_SPLITS.items():
        train = pd.read_csv(os.path.join(args.csv_dir, split, 'train.csv'))
        labels = cluster_split(train) if mode == 'cluster' else ligand_split(train)
        out = pd.DataFrame({'row': range(len(train)), 'split': labels})
        path = os.path.join(args.csv_dir, split, POSTHOC_VAL_ASSIGNMENT_CSV)
        out.to_csv(path, index=False)
        counts = out['split'].value_counts().to_dict()
        print(f'{split} ({mode}): {len(out)} rows {counts}; val ligands '
              f'{train.loc[out["split"] == "val", "ligand_id"].nunique()} -> {path}')


def cluster_split(
    train: pd.DataFrame,
) -> list[str]:
    """'val' / 'train' / 'straddling' per row from a seeded draw of whole MMseqs2 clusters."""
    clusters = sorted(set(train['tar_cluster']) | set(train['src_cluster']))
    random.Random(POSTHOC_VAL_SEED).shuffle(clusters)
    val = set(clusters[:int(POSTHOC_VAL_CLUSTER_FRACTION * len(clusters))])
    tar, src = train['tar_cluster'].isin(val), train['src_cluster'].isin(val)
    return ['val' if t and s else 'train' if not (t or s) else 'straddling' for t, s in zip(tar, src)]


def ligand_split(
    train: pd.DataFrame,
) -> list[str]:
    """'val' / 'train' per row from a seeded draw of whole ligand_ids."""
    ligands = sorted(train['ligand_id'].unique())
    random.Random(POSTHOC_VAL_SEED).shuffle(ligands)
    val = set(ligands[:int(POSTHOC_VAL_LIGAND_FRACTION * len(ligands))])
    return ['val' if lig in val else 'train' for lig in train['ligand_id']]


if __name__ == '__main__':
    main()
