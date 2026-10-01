"""
Convert squashed ligand_rmsd values in per-sample result CSVs back to Angstrom.

Between d6cc019 and the fix that added the `ligand_rmsd_loss` column, the validation loop wrote
the per-sample `ligand_rmsd` from the loss' per-sample dict. With `return_non_linear: true`
(set in every LocAlign config) that value is the loss term

    s = r / (1 + r / 10)          (r = ligand RMSD in Angstrom)

not r itself. A threshold `s < 4` therefore means `r < 6.67 A`. The exact inverse is

    r = s / (1 - s / 10)

`corr_rmsd` is not affected (it is stored raw). Baseline aligners (Dali, TM-align, US-align,
PLASMA, APoc, SoftAlign) are not affected: their ligand_rmsd comes from precomputed metadata.

Scale detection (per file, or per `experiment` group when that column exists):
  - a `ligand_rmsd_loss` column is present          -> already Angstrom (new logging)
  - max(ligand_rmsd) >= 10                          -> Angstrom (s is always < 10)
  - max(ligand_rmsd) < 10 and >= MIN_ROWS values    -> squashed
  - otherwise                                       -> unknown (left unchanged unless --scale is given)
On the validation sets used here, raw files always have max ligand_rmsd > 80 A.

For each input CSV that needs conversion this writes `<stem>.ligand_rmsd_angstrom.csv` next to the
original and never overwrites an existing file:
  - `ligand_rmsd` is replaced by the Angstrom value and the logged value is kept as `ligand_rmsd_loss`;
  - `success_ligand_rmsd<k>` columns (all_experiments_with_success.csv) are recomputed with
    scripts/offline_metrics.evaluate_success;
  - when a directory holds an all_experiments_with_success.csv that was converted, a corrected
    `success_rates.ligand_rmsd_angstrom.csv` is written next to its success_rates.csv.

Usage:
    python scripts/unsquash_ligand_rmsd.py PATH [PATH ...] [--scale auto|squashed] [--dry-run]
PATH may be a CSV file or a directory (its *.csv files, non-recursive).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RMSD0 = 10.0
MIN_ROWS = 50
SUFFIX = ".ligand_rmsd_angstrom.csv"
THRESHOLDS = (1, 2, 4)


def squash(r):
    return r / (1 + r / RMSD0)


def unsquash(s):
    s = pd.to_numeric(pd.Series(s), errors="coerce").astype(float)
    if (s >= RMSD0).any():
        raise ValueError(f"squashed ligand_rmsd must be < {RMSD0}; got max {s.max():.3f}")
    return s / (1 - s / RMSD0)


def detect_scale(values, has_loss_column=False):
    """Return 'angstrom', 'squashed' or 'unknown' for a vector of logged ligand_rmsd values."""
    if has_loss_column:
        return "angstrom"
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(v) == 0:
        return "unknown"
    if v.max() >= RMSD0:
        return "angstrom"
    if len(v) >= MIN_ROWS:
        return "squashed"
    return "unknown"


def _groups(df):
    if "experiment" in df.columns:
        return list(df.groupby("experiment", sort=False).groups.items())
    return [(None, df.index)]


def to_angstrom(df, scale="auto"):
    """Return (converted_df, report). Converts squashed groups only; leaves the rest unchanged."""
    if "ligand_rmsd" not in df.columns:
        return df, []
    out = df.copy()
    out["ligand_rmsd"] = pd.to_numeric(out["ligand_rmsd"], errors="coerce")
    has_loss = "ligand_rmsd_loss" in out.columns
    report = []
    converted_any = False
    for name, idx in _groups(out):
        vals = out.loc[idx, "ligand_rmsd"]
        detected = detect_scale(vals, has_loss)
        use = detected if scale == "auto" else scale
        report.append((name, len(idx), float(vals.max()) if vals.notna().any() else np.nan, detected, use))
        if use == "squashed":
            if not has_loss:
                if "ligand_rmsd_loss" not in out.columns:
                    out["ligand_rmsd_loss"] = np.nan
                out.loc[idx, "ligand_rmsd_loss"] = vals
            out.loc[idx, "ligand_rmsd"] = unsquash(vals).values
            converted_any = True
    return (out if converted_any else df), report


def recompute_success(df):
    """Recompute success_ligand_rmsd<k> columns with offline_metrics.evaluate_success."""
    from scripts.offline_metrics import SUCCESS_CRITERIA, evaluate_success
    for c in ["corr_rmsd", "atom_type_fraction", "cath_degree"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for k in THRESHOLDS:
        col = f"success_ligand_rmsd<{k}"
        if col in df.columns:
            df[col] = df.apply(lambda row: evaluate_success(row, ligand_rmsd_threshold=k, criteria=SUCCESS_CRITERIA), axis=1)
    return df


def summarize(combined, column_order=None):
    """Same summary as offline_metrics.py / offline_metrics_apo.py (success_rates.csv)."""
    data = {}
    for metric in ["corr_rmsd", "ligand_rmsd", "atom_type_fraction"]:
        if metric in combined.columns:
            data[f"mean_{metric}"] = combined.groupby("experiment")[metric].mean().round(5)
    easy = combined[combined["cath_degree"] < 4]
    hard = combined[combined["cath_degree"] == 4]
    for k in THRESHOLDS:
        col = f"success_ligand_rmsd<{k}"
        data[f"success_cath<4_rmsd<{k}"] = easy.groupby("experiment")[col].mean().round(5)
        data[f"success_cath=4_rmsd<{k}"] = hard.groupby("experiment")[col].mean().round(5)
    summary = pd.DataFrame(data)
    summary.index.name = "experiment"
    if column_order is not None:
        summary = summary[[c for c in column_order if c in summary.columns]]
    return summary


def _write(df, path, dry_run, **kw):
    if path.exists():
        print(f"  SKIP (exists, not overwriting): {path}")
        return False
    if not dry_run:
        df.to_csv(path, **kw)
    print(f"  wrote {path}" + (" (dry run)" if dry_run else ""))
    return True


def process_file(path, scale, dry_run):
    if path.name.endswith(SUFFIX):
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        print(f"{path}: unreadable ({e})")
        return None
    if "ligand_rmsd" not in df.columns:
        return None
    first_col_is_index = df.columns[0].startswith("Unnamed: 0")
    out, report = to_angstrom(df, scale)
    for name, n, mx, detected, use in report:
        label = f"[{name}] " if name is not None else ""
        print(f"{path}: {label}n={n} max={mx:.2f} detected={detected} -> {'convert' if use == 'squashed' else 'keep'}")
    if out is df:
        return None
    if any(c.startswith("success_ligand_rmsd<") for c in out.columns):
        out = recompute_success(out)
    if first_col_is_index:
        out = out.drop(columns=[df.columns[0]])
    _write(out, path.with_name(path.stem + SUFFIX), dry_run, index=False)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--scale", choices=["auto", "squashed"], default="auto",
                    help="'auto' detects per file/experiment; 'squashed' forces conversion (e.g. small files).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    for p in map(Path, args.paths):
        files = sorted(p.glob("*.csv")) if p.is_dir() else [p]
        converted = {}
        for f in files:
            out = process_file(f, args.scale, args.dry_run)
            if out is not None:
                converted[f.name] = out
        combined = converted.get("all_experiments_with_success.csv")
        sr = (p if p.is_dir() else p.parent) / "success_rates.csv"
        if combined is not None and sr.exists():
            old = pd.read_csv(sr, index_col=0)
            new = summarize(combined, column_order=list(old.columns))
            _write(new, sr.with_name(sr.stem + SUFFIX), args.dry_run)
            print("  old success_rates (as logged):")
            print(old.to_string())
            print("  corrected success_rates (Angstrom):")
            print(new.to_string())


if __name__ == "__main__":
    main()
