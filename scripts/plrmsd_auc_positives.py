"""pLRMSD for the positive (same-ligand) test pairs, computed out-of-fold.

Reproduces notebooks/train_calibration_model.ipynb cells 0-2 (the pLRMSD
calibration model) and cell 5 (the eLRMSD quantile model) for one split. Unlike
the notebook, whose cross_val_predict call silently uses an ungrouped KFold(5),
the same explicit GroupKFold(5, shuffle=True, random_state=seed), grouped by
ligand, is used for the grid search and for cross_val_predict.

eLRMSD is also computed out-of-fold, because the eLRMSD model is trained on the
test-set LRMSD labels. As in the notebook it is fit on tar_ligand_n_atoms; as in
scripts/inference.py it is evaluated on src_ligand_n_atoms.

Writes, per split, to --out_dir: positives_cv_predictions_<split>.csv,
calibration_model_<split>.pkl and ligand_calibration_model_<split>.pkl (both fit
on all positives) and best_params_<split>.json.

Usage:
  python scripts/plrmsd_auc_positives.py --split homology \
      --baseline_csv ablation_dfs/homology_split/baseline.csv
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'aligner_dl'))
from utils.constants import (  # noqa: E402
    ELRMSD_CALIBRATION_PKL,
    ELRMSD_QUANTILE,
    PLRMSD_AUC_DIR,
    PLRMSD_BEST_PARAMS_JSON,
    PLRMSD_CALIBRATION_PKL,
    PLRMSD_FEATURES,
    PLRMSD_MONOTONIC,
    PLRMSD_POSITIVES_CSV,
    PLRMSD_TARGET,
    PLRMSD_TARGET_CUTOFF,
)

PARAM_GRID = {
    'l2_regularization': np.logspace(-2, 0, num=5),
    'learning_rate': [0.025, 0.05, 0.075, 0.1],
    'max_iter': [50, 100, 200, 300, 400, 500],
    'max_leaf_nodes': [7, 15, 31],
}
N_SPLITS = 5
OUTPUT_COLUMNS = [
    'ligand_id', 'tar_protein', 'tar_chain', 'src_protein', 'src_chain', 'cath_degree',
    'tar_ligand_n_atoms', 'src_ligand_n_atoms', 'ligand_rmsd',
    'embedding_similarity', 'entropy', 'corr_rmsd', 'radius_of_gyration',
    'normalized_embedding_similarity', 'radius_of_gyration_ang', 'perplexity',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, choices=['homology', 'ligand'])
    parser.add_argument('--baseline_csv', required=True)
    parser.add_argument('--out_dir', default=PLRMSD_AUC_DIR)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_jobs', type=int, default=-1)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    table = add_features(pd.read_csv(args.baseline_csv))
    X = table[PLRMSD_FEATURES]
    y = table[PLRMSD_TARGET].clip(0, PLRMSD_TARGET_CUTOFF)
    cv = GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=args.seed)
    folds = list(cv.split(X, y, table['ligand_id']))
    fold_id = np.empty(len(table), dtype=int)
    for k, (_, test_idx) in enumerate(folds):
        fold_id[test_idx] = k

    search = grid_search(plrmsd_regressor(), X, y, folds, args.n_jobs)
    best = search.best_params_
    print(f'[{args.split}] pLRMSD best params: {best} (grid CV spearman {search.best_score_:.4f})')
    plrmsd = np.clip(cross_val_predict(plrmsd_regressor(**best), X, y, cv=folds), 0, PLRMSD_TARGET_CUTOFF)

    Xl_fit = table[['tar_ligand_n_atoms']].to_numpy()
    Xl_eval = table[['src_ligand_n_atoms']].to_numpy()
    search_l = grid_search(elrmsd_regressor(), Xl_fit, y, folds, args.n_jobs)
    best_l = search_l.best_params_
    print(f'[{args.split}] eLRMSD best params: {best_l} (grid CV spearman {search_l.best_score_:.4f})')
    elrmsd = np.empty(len(table))
    for train_idx, test_idx in folds:
        model = elrmsd_regressor(**best_l).fit(Xl_fit[train_idx], y.iloc[train_idx])
        elrmsd[test_idx] = model.predict(Xl_eval[test_idx])

    out = table[OUTPUT_COLUMNS].copy()
    out.insert(0, 'split', args.split)
    out['cv_fold'] = fold_id
    out['pLRMSD'] = plrmsd
    out['eLRMSD'] = elrmsd
    out['normalized_pLRMSD'] = plrmsd / elrmsd
    out_csv = os.path.join(args.out_dir, PLRMSD_POSITIVES_CSV.format(split=args.split))
    out.to_csv(out_csv, index=False)
    rho = spearmanr(plrmsd, y)[0]
    print(f'[{args.split}] n={len(out)} out-of-fold spearman(pLRMSD, clipped LRMSD)={rho:.4f} -> {out_csv}')

    with open(os.path.join(args.out_dir, PLRMSD_CALIBRATION_PKL.format(split=args.split)), 'wb') as f:
        pickle.dump(plrmsd_regressor(**best).fit(X, y), f)
    with open(os.path.join(args.out_dir, ELRMSD_CALIBRATION_PKL.format(split=args.split)), 'wb') as f:
        pickle.dump(elrmsd_regressor(**best_l).fit(Xl_fit, y), f)
    with open(os.path.join(args.out_dir, PLRMSD_BEST_PARAMS_JSON.format(split=args.split)), 'w') as f:
        json.dump({
            'seed': args.seed,
            'n_splits': N_SPLITS,
            'n_positives': int(len(out)),
            'baseline_csv': args.baseline_csv,
            'plrmsd_best_params': jsonable(best),
            'plrmsd_grid_cv_spearman': float(search.best_score_),
            'elrmsd_best_params': jsonable(best_l),
            'elrmsd_grid_cv_spearman': float(search_l.best_score_),
            'oof_spearman_plrmsd_lrmsd': float(rho),
        }, f, indent=2)


def first_n_atoms(x) -> int:
    """Number of atoms of the first ligand: '[40]' or '[40, 41]' -> 40."""
    if isinstance(x, (int, np.integer)):
        return int(x)
    return int(str(x).strip()[1:-1].split(',')[0])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the calibration features (notebook cell 0).

    Expects the raw model outputs embedding_similarity, entropy, corr_rmsd and
    radius_of_gyration, and converts the ligand atom-count columns to int.
    """
    df = df.copy()
    df['normalized_embedding_similarity'] = df['embedding_similarity'] / (df['entropy'] * np.log(400))
    df['perplexity'] = np.exp(df['entropy'] * np.log(400))
    df['radius_of_gyration_ang'] = df['radius_of_gyration'] * 1.3 * df['perplexity'] ** 0.4
    for col in ['tar_ligand_n_atoms', 'src_ligand_n_atoms']:
        if col in df.columns:
            df[col] = df[col].map(first_n_atoms)
    return df


def spearman_scorer(
    y_true,
    y_pred,
) -> float:
    return spearmanr(y_true, y_pred)[0]


def grid_search(
    estimator,
    X,
    y,
    cv,
    n_jobs: int,
) -> GridSearchCV:
    """Grid search maximizing Spearman correlation over the fixed folds `cv`."""
    search = GridSearchCV(
        estimator=estimator,
        param_grid=PARAM_GRID,
        cv=cv,
        scoring=make_scorer(spearman_scorer, greater_is_better=True),
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X, y)
    return search


def plrmsd_regressor(**params) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss='absolute_error',
        interaction_cst='no_interactions',
        monotonic_cst=np.array(PLRMSD_MONOTONIC),
        **params,
    )


def elrmsd_regressor(**params) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss='quantile',
        quantile=ELRMSD_QUANTILE,
        monotonic_cst=np.array([1]),
        max_bins=128,
        **params,
    )


def jsonable(d: dict) -> dict:
    return {k: (int(v) if isinstance(v, (int, np.integer)) else float(v)) for k, v in d.items()}


if __name__ == '__main__':
    main()
