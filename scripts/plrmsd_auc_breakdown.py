"""Breakdown of the pLRMSD AUC (T4, PR #12) by class: distribution of pLRMSD and
normalized pLRMSD, fraction of pairs at the pLRMSD cap, and AUC within the below-cap subset.

Inputs per split (written by scripts/plrmsd_auc_positives.py / plrmsd_auc_score.py):
  results/plrmsd_auc/positives_cv_predictions_<split>.csv  (out-of-fold, GroupKFold by ligand)
  results/plrmsd_auc/negatives_<split>.csv                 (scored by the all-positives model)

They are read from --in_dir if it contains them, otherwise from git ref --ref
(default origin/rebuttal2/t4-plrmsd-auc) via `git show`.

Output: results/plrmsd_auc/breakdown.csv, one row per split x class x subset
(subset = all | below_cap, below_cap = pLRMSD < 9.99), plus AUC rows for the below-cap subset.
"""
import argparse
import io
import os
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CAP = 9.99


def load(name, in_dir, ref):
    path = os.path.join(in_dir, name)
    if os.path.exists(path):
        return pd.read_csv(path)
    out = subprocess.run(['git', 'show', f'{ref}:results/plrmsd_auc/{name}'],
                         check=True, capture_output=True, text=True).stdout
    return pd.read_csv(io.StringIO(out))


def stats(x):
    q1, med, q3 = np.percentile(x, [25, 50, 75]) if len(x) else (np.nan,) * 3
    return med, q1, q3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', default='results/plrmsd_auc')
    ap.add_argument('--ref', default='origin/rebuttal2/t4-plrmsd-auc')
    ap.add_argument('--out', default='results/plrmsd_auc/breakdown.csv')
    args = ap.parse_args()

    rows = []
    for split in ['homology', 'ligand']:
        pos = load(f'positives_cv_predictions_{split}.csv', args.in_dir, args.ref)
        neg = load(f'negatives_{split}.csv', args.in_dir, args.ref)
        for cls, df in [('positive', pos), ('negative', neg)]:
            at_cap = df['pLRMSD'] >= CAP
            for subset, d in [('all', df), ('below_cap', df[~at_cap])]:
                p_med, p_q1, p_q3 = stats(d['pLRMSD'])
                n_med, n_q1, n_q3 = stats(d['normalized_pLRMSD'])
                rows.append({'split': split, 'class': cls, 'subset': subset, 'metric': 'distribution',
                             'n': len(d), 'n_total_class': len(df), 'frac_at_cap': at_cap.mean(),
                             'pLRMSD_median': p_med, 'pLRMSD_q1': p_q1, 'pLRMSD_q3': p_q3,
                             'norm_pLRMSD_median': n_med, 'norm_pLRMSD_q1': n_q1, 'norm_pLRMSD_q3': n_q3})
        # AUC within the below-cap subset (positive = same ligand, score = -pLRMSD / -normalized)
        p, n = pos[pos['pLRMSD'] < CAP], neg[neg['pLRMSD'] < CAP]
        y = np.r_[np.ones(len(p)), np.zeros(len(n))]
        for score in ['pLRMSD', 'normalized_pLRMSD']:
            s = -np.r_[p[score].to_numpy(), n[score].to_numpy()]
            rows.append({'split': split, 'class': 'pos_vs_neg', 'subset': 'below_cap',
                         'metric': f'auc_-{score}', 'n': len(y), 'n_pos': len(p), 'n_neg': len(n),
                         'auc': roc_auc_score(y, s)})
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False, float_format='%.4f')
    print(out.to_string())


if __name__ == '__main__':
    main()
