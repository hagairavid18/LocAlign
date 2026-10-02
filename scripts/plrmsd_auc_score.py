"""Score the negatives with the split-specific calibration models and compute the
AUC-ROC of pLRMSD for positive (same-ligand) vs negative (no shared ligand) pairs.

Inputs per split, read from --out_dir:
  positives_cv_predictions_<split>.csv   (plrmsd_auc_positives.py)
  negatives_input_<split>.csv            (plrmsd_auc_sample_negatives.py)
  calibration_model_<split>.pkl, ligand_calibration_model_<split>.pkl,
  best_params_<split>.json               (plrmsd_auc_positives.py)
  the inference_results.csv of the negatives run (plrmsd_auc_run_inference.py)

Outputs: negatives_<split>.csv, failed_negatives_<split>.csv (if any), auc.csv,
breakdown.csv and roc_<split>.png, all in --out_dir.

Label 1 = positive, 0 = negative. The scores are -normalized_pLRMSD (main) and
-pLRMSD, each with negatives scored by the all-positives model and, as a robustness
check, fold-matched (suffix _foldmatched). Rows with positives=LRMSD<=4A use only the
positives with ligand RMSD <= AUC_LRMSD_SUCCESS_CUTOFF against all negatives. The 95% CI is a stratified
bootstrap over pairs.

Scores are rounded to AUC_SCORE_DECIMALS decimals before every AUC, so that the
many pairs saturated at the pLRMSD cap count as ties (0.5) instead of being ordered
by floating point noise.

breakdown.csv gives, per split and class, the distribution of pLRMSD and normalized
pLRMSD, the fraction of pairs at the cap (pLRMSD >= PLRMSD_CAP_THRESHOLD), and the
AUC within the below-cap subset.
"""
import argparse
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from plrmsd_auc_positives import add_features, elrmsd_regressor, plrmsd_regressor  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'aligner_dl'))
from utils.constants import (  # noqa: E402
    AUC_BOOTSTRAP_RESAMPLES,
    AUC_LRMSD_SUCCESS_CUTOFF,
    AUC_SCORE_DECIMALS,
    ELRMSD_CALIBRATION_PKL,
    INFERENCE_RAW_COLUMNS,
    PAIR_KEY_COLUMNS,
    PLRMSD_AUC_CSV,
    PLRMSD_AUC_DIR,
    PLRMSD_BEST_PARAMS_JSON,
    PLRMSD_BREAKDOWN_CSV,
    PLRMSD_CALIBRATION_PKL,
    PLRMSD_CAP_THRESHOLD,
    PLRMSD_FAILED_NEGATIVES_CSV,
    PLRMSD_FEATURES,
    PLRMSD_NEGATIVES_CSV,
    PLRMSD_NEGATIVES_INPUT_CSV,
    PLRMSD_POSITIVES_CSV,
    PLRMSD_ROC_PNG,
    PLRMSD_TARGET_CUTOFF,
)

SCORES = [
    ('normalized_pLRMSD', 'BLUE', '-', 'normalized pLRMSD'),
    ('pLRMSD', 'ORANGE', '-', 'pLRMSD'),
    ('normalized_pLRMSD_foldmatched', 'BLUE', ':', 'normalized, fold-matched'),
    ('pLRMSD_foldmatched', 'ORANGE', ':', 'pLRMSD, fold-matched'),
]
COLORS = {'BLUE': '#2a78d6', 'ORANGE': '#eb6834'}
INK, MUTED = '#1f1f1f', '#8a8a8a'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default=PLRMSD_AUC_DIR)
    parser.add_argument('--raw_homology', required=True)
    parser.add_argument('--raw_ligand', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    auc_rows, breakdown_rows = [], []
    for split, raw_csv in [('homology', args.raw_homology), ('ligand', args.raw_ligand)]:
        neg, failed = score_negatives(split, raw_csv, args.out_dir)
        pos = pd.read_csv(os.path.join(args.out_dir, PLRMSD_POSITIVES_CSV.format(split=split)))
        for c in ['pLRMSD', 'normalized_pLRMSD']:
            pos[f'{c}_foldmatched'] = pos[c]
        rows = auc_table(split, pos, neg, len(failed), rng, args.out_dir)
        auc_rows.extend(rows)
        breakdown_rows.extend(breakdown_table(split, pos, neg))
    pd.DataFrame(auc_rows).to_csv(os.path.join(args.out_dir, PLRMSD_AUC_CSV), index=False, float_format='%.4f')
    breakdown = pd.DataFrame(breakdown_rows)
    breakdown.to_csv(os.path.join(args.out_dir, PLRMSD_BREAKDOWN_CSV), index=False, float_format='%.4f')
    print(breakdown.to_string())


def score_negatives(
    split: str,
    raw_csv: str,
    out_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pLRMSD, eLRMSD and normalized pLRMSD of the negatives from the inference outputs.

    Each negative is scored twice: by the models fit on all positives, and, as a
    robustness check, by the models of a random CV fold (seed from best_params), so
    that positives (scored out-of-fold) and negatives are scored by the same mixture
    of models. HistGradientBoosting without early stopping is deterministic, so the
    refitted fold models reproduce the stored out-of-fold predictions (asserted).
    Returns the scored negatives and the rows for which inference failed.
    """
    neg = pd.read_csv(
        os.path.join(out_dir, PLRMSD_NEGATIVES_INPUT_CSV.format(split=split)),
        dtype={'tar_chain': str, 'src_chain': str},
    )
    raw = pd.read_csv(raw_csv, dtype={'tar_chain': str, 'src_chain': str})
    raw = raw[PAIR_KEY_COLUMNS + list(INFERENCE_RAW_COLUMNS)].rename(columns=INFERENCE_RAW_COLUMNS)
    raw = raw.dropna(subset=list(INFERENCE_RAW_COLUMNS.values())).drop_duplicates(subset=PAIR_KEY_COLUMNS)
    df = neg.merge(raw, on=PAIR_KEY_COLUMNS, how='left')
    failed = df[df['embedding_similarity'].isna()]
    ok = add_features(df[df['embedding_similarity'].notna()])
    with open(os.path.join(out_dir, PLRMSD_CALIBRATION_PKL.format(split=split)), 'rb') as f:
        cal = pickle.load(f)
    with open(os.path.join(out_dir, ELRMSD_CALIBRATION_PKL.format(split=split)), 'rb') as f:
        lcal = pickle.load(f)
    ok['pLRMSD'] = np.clip(cal.predict(ok[PLRMSD_FEATURES]), 0, PLRMSD_TARGET_CUTOFF)
    ok['eLRMSD'] = lcal.predict(ok[['src_ligand_n_atoms']].to_numpy())
    ok['normalized_pLRMSD'] = ok['pLRMSD'] / ok['eLRMSD']

    pos = pd.read_csv(os.path.join(out_dir, PLRMSD_POSITIVES_CSV.format(split=split)))
    with open(os.path.join(out_dir, PLRMSD_BEST_PARAMS_JSON.format(split=split))) as f:
        bp = json.load(f)
    y = pos['ligand_rmsd'].clip(0, PLRMSD_TARGET_CUTOFF)
    rng = np.random.default_rng(bp['seed'])
    ok['cv_fold_matched'] = rng.integers(bp['n_splits'], size=len(ok))
    ok['pLRMSD_foldmatched'] = np.nan
    ok['eLRMSD_foldmatched'] = np.nan
    for k in range(bp['n_splits']):
        tr, te = pos['cv_fold'] != k, pos['cv_fold'] == k
        mp = plrmsd_regressor(**bp['plrmsd_best_params']).fit(pos.loc[tr, PLRMSD_FEATURES], y[tr])
        ml = elrmsd_regressor(**bp['elrmsd_best_params']).fit(pos.loc[tr, ['tar_ligand_n_atoms']].to_numpy(), y[tr])
        assert np.allclose(
            np.clip(mp.predict(pos.loc[te, PLRMSD_FEATURES]), 0, PLRMSD_TARGET_CUTOFF),
            pos.loc[te, 'pLRMSD'],
        )
        assert np.allclose(ml.predict(pos.loc[te, ['src_ligand_n_atoms']].to_numpy()), pos.loc[te, 'eLRMSD'])
        m = ok['cv_fold_matched'] == k
        ok.loc[m, 'pLRMSD_foldmatched'] = np.clip(mp.predict(ok.loc[m, PLRMSD_FEATURES]), 0, PLRMSD_TARGET_CUTOFF)
        ok.loc[m, 'eLRMSD_foldmatched'] = ml.predict(ok.loc[m, ['src_ligand_n_atoms']].to_numpy())
    ok['normalized_pLRMSD_foldmatched'] = ok['pLRMSD_foldmatched'] / ok['eLRMSD_foldmatched']
    ok.insert(0, 'split', split)
    ok.to_csv(os.path.join(out_dir, PLRMSD_NEGATIVES_CSV.format(split=split)), index=False)
    if len(failed):
        failed[PAIR_KEY_COLUMNS + ['tar_ligand', 'src_ligand']].to_csv(
            os.path.join(out_dir, PLRMSD_FAILED_NEGATIVES_CSV.format(split=split)),
            index=False,
        )
    return ok, failed


def auc_score(
    y: np.ndarray,
    s: np.ndarray,
) -> float:
    """AUC-ROC of scores `s` rounded to AUC_SCORE_DECIMALS decimals, so that plateau ties count 0.5."""
    return roc_auc_score(y, np.round(s, AUC_SCORE_DECIMALS))


def boot_ci(
    y: np.ndarray,
    s: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """95% percentile CI of the AUC from a stratified bootstrap over pairs."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(AUC_BOOTSTRAP_RESAMPLES):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(auc_score(y[idx], s[idx]))
    return np.percentile(vals, [2.5, 97.5])


def auc_table(
    split: str,
    pos: pd.DataFrame,
    neg: pd.DataFrame,
    n_failed: int,
    rng: np.random.Generator,
    out_dir: str,
) -> list[dict]:
    """AUC rows (all positives and LRMSD<=cutoff positives, for every score) and the ROC plot of one split."""
    rows = []
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for score, color, ls, label in SCORES:
        s = -np.r_[pos[score].to_numpy(), neg[score].to_numpy()]
        auc = auc_score(y, s)
        lo, hi = boot_ci(y, s, rng)
        rows.append({
            'split': split, 'positives': 'all', 'score': f'-{score}', 'auc': auc,
            'ci95_low': lo, 'ci95_high': hi, 'n_pos': len(pos), 'n_neg': len(neg), 'n_neg_failed': n_failed,
            'median_score_pos': float(np.median(pos[score])), 'median_score_neg': float(np.median(neg[score])),
        })
        fpr, tpr, _ = roc_curve(y, np.round(s, AUC_SCORE_DECIMALS))
        ax.plot(fpr, tpr, color=COLORS[color], lw=2, ls=ls, label=f'{label}  AUC {auc:.3f}')
        print(f'[{split}] -{score}: AUC={auc:.4f} [{lo:.4f}, {hi:.4f}] n_pos={len(pos)} n_neg={len(neg)} failed={n_failed}')
        ps = pos[pos['ligand_rmsd'] <= AUC_LRMSD_SUCCESS_CUTOFF]
        ys = np.r_[np.ones(len(ps)), np.zeros(len(neg))]
        ss = -np.r_[ps[score].to_numpy(), neg[score].to_numpy()]
        lo_s, hi_s = boot_ci(ys, ss, rng)
        rows.append({
            'split': split, 'positives': 'LRMSD<=4A', 'score': f'-{score}', 'auc': auc_score(ys, ss),
            'ci95_low': lo_s, 'ci95_high': hi_s, 'n_pos': len(ps), 'n_neg': len(neg), 'n_neg_failed': n_failed,
            'median_score_pos': float(np.median(ps[score])), 'median_score_neg': float(np.median(neg[score])),
        })
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('False positive rate', color=INK)
    ax.set_ylabel('True positive rate', color=INK)
    ax.set_title(
        f'{split.capitalize()} split: same-ligand vs random pairs\n{len(pos)} positives, {len(neg)} negatives',
        color=INK,
        fontsize=11,
    )
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.grid(color='#e6e6e6', lw=0.6)
    ax.legend(loc='lower right', frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, PLRMSD_ROC_PNG.format(split=split)), dpi=200)
    plt.close(fig)
    return rows


def quartiles(
    x: pd.Series,
) -> tuple[float, float, float]:
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return med, q1, q3


def breakdown_table(
    split: str,
    pos: pd.DataFrame,
    neg: pd.DataFrame,
) -> list[dict]:
    """Distribution rows per class (all pairs and below-cap pairs) and below-cap AUC rows of one split."""
    rows = []
    for cls, df in [('positive', pos), ('negative', neg)]:
        at_cap = df['pLRMSD'] >= PLRMSD_CAP_THRESHOLD
        for subset, d in [('all', df), ('below_cap', df[~at_cap])]:
            p_med, p_q1, p_q3 = quartiles(d['pLRMSD'])
            n_med, n_q1, n_q3 = quartiles(d['normalized_pLRMSD'])
            rows.append({
                'split': split, 'class': cls, 'subset': subset, 'metric': 'distribution',
                'n': len(d), 'n_total_class': len(df), 'frac_at_cap': at_cap.mean(),
                'pLRMSD_median': p_med, 'pLRMSD_q1': p_q1, 'pLRMSD_q3': p_q3,
                'norm_pLRMSD_median': n_med, 'norm_pLRMSD_q1': n_q1, 'norm_pLRMSD_q3': n_q3,
            })
    p = pos[pos['pLRMSD'] < PLRMSD_CAP_THRESHOLD]
    n = neg[neg['pLRMSD'] < PLRMSD_CAP_THRESHOLD]
    y = np.r_[np.ones(len(p)), np.zeros(len(n))]
    for score in ['pLRMSD', 'normalized_pLRMSD']:
        s = -np.r_[p[score].to_numpy(), n[score].to_numpy()]
        rows.append({
            'split': split, 'class': 'pos_vs_neg', 'subset': 'below_cap', 'metric': f'auc_-{score}',
            'n': len(y), 'n_pos': len(p), 'n_neg': len(n), 'auc': auc_score(y, s),
        })
    return rows


if __name__ == '__main__':
    main()
