"""ROC curves of raw pLRMSD for same-ligand vs random pairs, one panel per split.

Reads the outputs of scripts/plrmsd_auc_score.py in --auc_dir (positives_cv_predictions_<split>.csv,
negatives_<split>.csv, auc.csv) and writes <stem>.pdf and <stem>.png (PLRMSD_ROC_FIGURE_DPI dpi)
into --out_dir. Nothing is refitted or re-scored.

The score is -pLRMSD. Solid line: all positives vs negatives; dashed line: positives with
ligand RMSD <= AUC_LRMSD_SUCCESS_CUTOFF vs negatives; dotted diagonal: random. Scores are
rounded to AUC_SCORE_DECIMALS decimals as in plrmsd_auc_score.py, so pairs at the pLRMSD cap
are ties (they count 0.5 in the AUC and form one diagonal segment of the curve).
The legend shows the AUC and 95% bootstrap CI from auc.csv; the AUC is recomputed here and
must match auc.csv (asserted to 4 decimals).

Usage
  python scripts/plrmsd_roc_figure.py
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'aligner_dl'))
from utils.constants import (  # noqa: E402
    AUC_LRMSD_SUCCESS_CUTOFF,
    AUC_SCORE_DECIMALS,
    PLRMSD_AUC_CSV,
    PLRMSD_AUC_DIR,
    PLRMSD_NEGATIVES_CSV,
    PLRMSD_POSITIVES_CSV,
    PLRMSD_ROC_FIGURE_DPI,
    PLRMSD_ROC_FIGURE_STEM,
)

SPLITS = [('homology', 'Homology split'), ('ligand', 'Ligand split')]
SERIES_COLOR = '#2a78d6'
INK, MUTED, GRID = '#1f1f1f', '#8a8a8a', '#e6e6e6'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--auc_dir', default=PLRMSD_AUC_DIR)
    ap.add_argument('--out_dir', default=PLRMSD_AUC_DIR)
    args = ap.parse_args()

    auc = pd.read_csv(os.path.join(args.auc_dir, PLRMSD_AUC_CSV))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
    for ax, (split, title) in zip(axes, SPLITS):
        pos = pd.read_csv(os.path.join(args.auc_dir, PLRMSD_POSITIVES_CSV.format(split=split)))
        neg = pd.read_csv(os.path.join(args.auc_dir, PLRMSD_NEGATIVES_CSV.format(split=split)))
        good = pos[pos['ligand_rmsd'] <= AUC_LRMSD_SUCCESS_CUTOFF]
        curves = [
            ('all', pos, '-', 'All positives'),
            (f'LRMSD<={AUC_LRMSD_SUCCESS_CUTOFF:g}A', good, '--', f'Positives with LRMSD ≤ {AUC_LRMSD_SUCCESS_CUTOFF:g} Å'),
        ]
        for positives, p, ls, label in curves:
            row = auc_row(
                auc,
                split,
                positives,
            )
            y = np.r_[np.ones(len(p)), np.zeros(len(neg))]
            s = np.round(-np.r_[p['pLRMSD'].to_numpy(), neg['pLRMSD'].to_numpy()], AUC_SCORE_DECIMALS)
            value = roc_auc_score(y, s)
            assert round(value, 4) == round(row['auc'], 4), (split, positives, value, row['auc'])
            assert (len(p), len(neg)) == (row['n_pos'], row['n_neg']), (split, positives)
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, color=SERIES_COLOR, lw=2, ls=ls,
                    label=f'{label}: AUC {row["auc"]:.3f} [{row["ci95_low"]:.3f}, {row["ci95_high"]:.3f}]')
            print(f'{split:8s} {positives:10s} AUC {row["auc"]:.4f} [{row["ci95_low"]:.4f}, {row["ci95_high"]:.4f}] '
                  f'n_pos {len(p)} n_neg {len(neg)}')
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=':', label='Random: AUC 0.5')
        style_axes(
            ax,
            f'{title}\n{len(pos)} positives, {len(neg)} negatives',
        )
    fig.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        path = os.path.join(args.out_dir, f'{PLRMSD_ROC_FIGURE_STEM}.{ext}')
        fig.savefig(path, dpi=PLRMSD_ROC_FIGURE_DPI)
        print(f'wrote {path}')
    plt.close(fig)


def auc_row(
    auc: pd.DataFrame,
    split: str,
    positives: str,
) -> pd.Series:
    """The auc.csv row of raw pLRMSD (score -pLRMSD) for one split and positive set."""
    rows = auc[(auc['split'] == split) & (auc['positives'] == positives) & (auc['score'] == '-pLRMSD')]
    assert len(rows) == 1, (split, positives, len(rows))
    return rows.iloc[0]


def style_axes(
    ax: plt.Axes,
    title: str,
) -> None:
    """Square unit axes, recessive grid, no top/right spines, legend in the lower right."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('False positive rate', color=INK)
    ax.set_ylabel('True positive rate', color=INK)
    ax.set_title(title, color=INK, fontsize=11)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.legend(loc='lower right', frameon=False, fontsize=8, labelcolor=INK)


if __name__ == '__main__':
    main()
