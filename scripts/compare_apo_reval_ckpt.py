"""
Compare two apo/holo re-evaluation runs of the ligand-split test pairs, e.g. the run made
with the homology-split checkpoint (ablation_dfs/apo_reanalysis/ligand_split) and the run
made with the ligand-split checkpoint (ablation_dfs/apo_reanalysis/ligand_split_ligckpt).

Inputs are the all_experiments_with_success.csv files written by
scripts/offline_metrics_apo.py. Per subset it reports n, composite success
(corr_rmsd<2 & ligand_rmsd<4 & atom_type_fraction>0.5, NaN counted as failure),
ligand-RMSD-only success (ligand_rmsd<4), and both split by fold
(cath_degree<4 = different fold, cath_degree==4 = same fold).

Ligand-RMSD scale
-----------------
Since commit d6cc019 the per-sample `ligand_rmsd` written by the Lightning validation loop
(per_sample_results_*.csv) is taken from the loss' per-sample dict. With
`return_non_linear: true` in the loss config that value is the squashed loss term
    s = r / (1 + r / 10)          (r = ligand RMSD in Angstrom, s < 10)
not r itself. Thresholding s < 4 is equivalent to r < 6.67 A. The per-pair tables produced
before that commit (e.g. ablation_dfs/ligand_split/baseline.csv) hold raw r.
This script therefore reports every success rate twice:
    lrmsd_scale = "as_logged": the value in the CSV, i.e. what offline_metrics_apo.py thresholds
    lrmsd_scale = "raw":       r = s / (1 - s / 10), the exact inverse, thresholded at 4 A
Pass --raw_input for CSVs that already hold raw RMSD.

Optionally (--reference) it checks the holo_holo_subset run of --new against a reference
per-pair table (raw RMSD, e.g. ablation_dfs/ligand_split/baseline.csv) on the same pairs.

Usage:
    python scripts/compare_apo_reval_ckpt.py \
        --old ablation_dfs/apo_reanalysis/ligand_split/all_experiments_with_success.csv \
        --new ablation_dfs/apo_reanalysis/ligand_split_ligckpt/all_experiments_with_success.csv \
        --reference ablation_dfs/ligand_split/baseline.csv \
        --out ablation_dfs/apo_reanalysis/ligand_split_ligckpt/old_vs_new_ckpt.csv
"""
import argparse

import numpy as np
import pandas as pd

SUBSETS = ["holo_holo_subset", "apo_apo", "apo_holo"]
KEY = ["ligand_id", "src_protein", "src_chain", "tar_protein", "tar_chain"]
RMSD0 = 10.0  # LigandLoss default rmsd0


def unsquash(s: pd.Series) -> pd.Series:
    """Invert s = r / (1 + r / RMSD0)."""
    s = pd.to_numeric(s, errors="coerce")
    return s / (1.0 - s / RMSD0)


def summarize(df: pd.DataFrame, lrmsd: pd.Series) -> dict:
    cath = pd.to_numeric(df["cath_degree"], errors="coerce")
    c = pd.to_numeric(df["corr_rmsd"], errors="coerce")
    a = pd.to_numeric(df["atom_type_fraction"], errors="coerce")
    diff, same = cath < 4, cath == 4
    lr = (lrmsd < 4).fillna(False).astype(bool)
    comp = ((c < 2) & (lrmsd < 4) & (a > 0.5)).fillna(False).astype(bool)
    return {
        "n": len(df),
        "n_diff_fold": int(diff.sum()),
        "n_same_fold": int(same.sum()),
        "composite": comp.mean(),
        "composite_diff_fold": comp[diff].mean(),
        "composite_same_fold": comp[same].mean(),
        "lrmsd_only": lr.mean(),
        "lrmsd_only_diff_fold": lr[diff].mean(),
        "lrmsd_only_same_fold": lr[same].mean(),
        "median_ligand_rmsd": lrmsd.median(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--old_label", default="homology_ckpt")
    ap.add_argument("--new_label", default="ligand_ckpt")
    ap.add_argument("--raw_input", action="store_true", help="Input ligand_rmsd is already raw (no inversion).")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    old, new = pd.read_csv(args.old), pd.read_csv(args.new)
    rows = []
    for scale in ["as_logged", "raw"]:
        for s in SUBSETS:
            for label, df in [(args.old_label, old), (args.new_label, new)]:
                sub = df[df["experiment"] == s]
                logged = pd.to_numeric(sub["ligand_rmsd"], errors="coerce")
                lrmsd = logged if (scale == "as_logged" or args.raw_input) else unsquash(logged)
                rows.append({"lrmsd_scale": scale, "subset": s, "checkpoint": label, **summarize(sub, lrmsd)})
    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(table)
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"Saved to {args.out}")

    if args.reference:
        ref = pd.read_csv(args.reference)
        n_dup = int(ref.duplicated(KEY).sum())
        ref = ref.drop_duplicates(KEY)  # the val table lists a few identical pairs twice
        hh = new[new["experiment"] == "holo_holo_subset"]
        m = hh.merge(ref, on=KEY, suffixes=("_apo", "_ref"))
        print(f"\nReference check ({args.new_label} holo_holo_subset vs {args.reference}): "
              f"{len(hh)} rows, {len(m)} matched ({n_dup} duplicate keys dropped from reference)")
        m["ligand_rmsd_apo_raw"] = m["ligand_rmsd_apo"] if args.raw_input else unsquash(m["ligand_rmsd_apo"])
        for col_apo, col_ref in [("ligand_rmsd_apo", "ligand_rmsd_ref"), ("ligand_rmsd_apo_raw", "ligand_rmsd_ref"),
                                 ("corr_rmsd_apo", "corr_rmsd_ref"),
                                 ("atom_type_fraction_apo", "atom_type_fraction_ref")]:
            x = pd.to_numeric(m[col_apo], errors="coerce")
            y = pd.to_numeric(m[col_ref], errors="coerce")
            ok = x.notna() & y.notna()
            d = (x[ok] - y[ok]).abs()
            print(f"  {col_apo} vs {col_ref}: median|diff|={d.median():.4g} "
                  f"frac|diff|<0.01={np.mean(d < 0.01):.4f} pearson={np.corrcoef(x[ok], y[ok])[0, 1]:.4f}")
        apo_m = m.rename(columns={f"{c}_apo": c for c in ["corr_rmsd", "atom_type_fraction", "cath_degree"]})
        ref_m = m.rename(columns={f"{c}_ref": c for c in ["corr_rmsd", "atom_type_fraction", "cath_degree"]})
        fmt = lambda d: {k: round(float(v), 4) for k, v in d.items()}
        print("  same pairs, run (raw LRMSD) :", fmt(summarize(apo_m, m["ligand_rmsd_apo_raw"])))
        print("  same pairs, reference       :", fmt(summarize(ref_m, pd.to_numeric(m["ligand_rmsd_ref"], errors="coerce"))))


if __name__ == "__main__":
    main()
