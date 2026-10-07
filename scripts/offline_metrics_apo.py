"""
Success-rate aggregation for the apo/holo reanalysis tables.
Reuses process_experiment/evaluate_success from scripts/offline_metrics.py UNCHANGED
(so the paper's existing success_rates.csv generation is untouched), just pointed at a
new sibling directory.

Expects, per split, the 3 per_sample_results_*.csv outputs from the apo-reval Lightning
runs copied/renamed to:
    ablation_dfs/apo_reanalysis/{homology_split,ligand_split}/{holo_holo_subset,apo_apo,apo_holo}.csv
Only these three files are read; the other CSVs in the directory (this script's own outputs,
older copies of the inputs) are ignored. Each input must hold ligand_rmsd in Angstrom: a column
whose values are all below 10 is the loss-scale r / (1 + r / 10), which is always below 10, and is
rejected.

Usage:
    python scripts/offline_metrics_apo.py
"""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "aligner_dl"))

from scripts.offline_metrics import SUCCESS_CRITERIA, process_experiment  # noqa: E402
from utils.constants import APO_REANALYSIS_SPLIT_DIRS  # noqa: E402

ABLATION_DIRS = APO_REANALYSIS_SPLIT_DIRS
EXPERIMENTS = ("holo_holo_subset", "apo_apo", "apo_holo")
LOSS_SCALE_LIMIT = 10.0
MIN_PAIRS_FOR_UNIT_CHECK = 50


def experiment_files(
    ablation_dir: str,
) -> list[Path]:
    """The input CSVs of the apo/holo experiments that exist in `ablation_dir`, in EXPERIMENTS order."""
    return [Path(ablation_dir) / f"{name}.csv" for name in EXPERIMENTS if (Path(ablation_dir) / f"{name}.csv").exists()]


def check_ligand_rmsd_unit(
    csv_path: Path,
) -> None:
    """Raise if the ligand_rmsd of `csv_path` looks like the loss-scale value instead of Angstrom."""
    values = pd.read_csv(csv_path, usecols=["ligand_rmsd"]).ligand_rmsd.dropna()
    if len(values) >= MIN_PAIRS_FOR_UNIT_CHECK and values.max() < LOSS_SCALE_LIMIT:
        raise ValueError(
            f"{csv_path}: ligand_rmsd is below {LOSS_SCALE_LIMIT:g} for all {len(values)} pairs (maximum {values.max():.2f}); "
            "that is the loss-scale value r / (1 + r / 10), not Angstrom"
        )


def main() -> None:
    for ablation_dir in ABLATION_DIRS:
        dir_name = Path(ablation_dir).name
        csv_files = experiment_files(ablation_dir)
        if not csv_files:
            print(f"None of {', '.join(f'{name}.csv' for name in EXPERIMENTS)} found in {ablation_dir}")
            continue

        all_experiments = []
        for csv_path in csv_files:
            experiment_name = csv_path.stem
            check_ligand_rmsd_unit(csv_path)
            all_experiments.append(process_experiment(csv_path, experiment_name, SUCCESS_CRITERIA))

        if not all_experiments:
            print(f"No experiments were successfully processed in {dir_name}.")
            continue

        combined_df = pd.concat(all_experiments, ignore_index=True)
        combined_df.to_csv(os.path.join(ablation_dir, "all_experiments_with_success.csv"), index=False)

        summary_data = {}
        for metric in ["corr_rmsd", "ligand_rmsd", "atom_type_fraction"]:
            if metric in combined_df.columns:
                summary_data[f"mean_{metric}"] = combined_df.groupby("experiment")[metric].mean().round(5)

        if "cath_degree" in combined_df.columns:
            easy_df = combined_df[combined_df["cath_degree"] < 4]
            hard_df = combined_df[combined_df["cath_degree"] == 4]
            for threshold in [1, 2, 4]:
                col = f"success_ligand_rmsd<{threshold}"
                summary_data[f"success_cath<4_rmsd<{threshold}"] = easy_df.groupby("experiment")[col].mean().round(5)
                summary_data[f"success_cath=4_rmsd<{threshold}"] = hard_df.groupby("experiment")[col].mean().round(5)
        else:
            print("Warning: 'cath_degree' column not found. Cannot compute filtered success rates.")

        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(ablation_dir, "success_rates.csv")
        summary_df.to_csv(summary_path)
        print(f"\nSuccess rates for {dir_name}:")
        print(summary_df)
        print(f"Saved to {summary_path}")


if __name__ == "__main__":
    main()
