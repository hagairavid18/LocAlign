"""
Success-rate aggregation for the reviewer-comment-#1 apo/holo reanalysis tables.
Reuses process_experiment/evaluate_success from scripts/offline_metrics.py UNCHANGED
(so the paper's existing success_rates.csv generation is untouched), just pointed at a
new sibling directory. See the approved plan at
/home/iscb/wolfson/hagairavid/.claude/plans/gleaming-greeting-stallman.md, section 7.

Expects, per split, the 3 per_sample_results_*.csv outputs from the apo-reval Lightning
runs copied/renamed to:
    ablation_dfs/apo_reanalysis/{homology_split,ligand_split}/{holo_holo_subset,apo_apo,apo_holo}.csv

Usage:
    python scripts/offline_metrics_apo.py
"""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.getcwd())

from scripts.offline_metrics import SUCCESS_CRITERIA, process_experiment

ABLATION_DIRS = [
    "/home/iscb/wolfson/hagairavid/LocAlign/ablation_dfs/apo_reanalysis/homology_split",
    "/home/iscb/wolfson/hagairavid/LocAlign/ablation_dfs/apo_reanalysis/ligand_split",
]


def main() -> None:
    for ablation_dir in ABLATION_DIRS:
        dir_name = Path(ablation_dir).name
        csv_files = sorted(Path(ablation_dir).glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {ablation_dir}")
            continue

        all_experiments = []
        for csv_path in csv_files:
            experiment_name = csv_path.stem
            try:
                all_experiments.append(process_experiment(csv_path, experiment_name, SUCCESS_CRITERIA))
            except Exception as e:
                print(f"Error processing {experiment_name}: {e}")

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
