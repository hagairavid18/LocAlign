import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "aligner_dl"))
from utils.constants import ABLATION_SPLIT_DIRS, SYMMETRY_COUNTS_PATH, SYMMETRY_RELAXED_LIGAND_RMSD  # noqa: E402

# Configuration: Success criteria thresholds
SUCCESS_CRITERIA = {
    'corr_rmsd': 2.0,
    'ligand_rmsd': 4.0,
    'atom_type_fraction': 0.5
}

# Directories containing experiment CSVs
ABLATION_DIRS = ABLATION_SPLIT_DIRS



def evaluate_success(row, ligand_rmsd_threshold=4.0, criteria=SUCCESS_CRITERIA):
    """
    Evaluate success criteria, using only ligand_rmsd for Dali/SoftAlign/TMalign/PLASMA.
    """
    exp = row.get('experiment')
    
    # For baseline aligners, check if ligand_rmsd is NaN
    if pd.isna(row['ligand_rmsd']):
        return False
    
    if exp in ('Dali', 'SoftAlign', 'TMalign', 'PLASMA', 'USAlign', 'USAlign_fns', 'USAlign_sns', 'APoc'):
        return row['ligand_rmsd'] < ligand_rmsd_threshold
    success = True
    success &= row['corr_rmsd'] < criteria['corr_rmsd']
    success &= row['ligand_rmsd'] < ligand_rmsd_threshold
    success &= row['atom_type_fraction'] > criteria['atom_type_fraction']
    return success


def process_experiment(
    csv_path,
    experiment_name,
    criteria=SUCCESS_CRITERIA,
    symmetric_ligands: set[str] | None = None,
):
    """
    Load experiment CSV and evaluate success for each row with multiple ligand_rmsd thresholds.
    
    Args:
        csv_path: Path to experiment CSV file
        experiment_name: Name of the experiment
        criteria: Success criteria dict
        
    Also adds `success_lrmsd_only<4` (ligand_rmsd < 4 alone, NaN counts as failure), the criterion
    used for the baseline aligners, so every experiment can be compared under it.

    If symmetric_ligands is given, also adds an UPPER BOUND on symmetry-corrected success: a pair
    with a symmetric ligand also counts as a success when the criterion holds at ligand_rmsd < 6
    instead of < 4 (`success_sym_upper_bound<4`, and `success_lrmsd_only_sym_upper_bound<4` for
    LRMSD-only). The RMSD is not recomputed under atom relabeling.

    Returns:
        DataFrame with success columns for different ligand_rmsd thresholds and experiment metadata
    """
    df = pd.read_csv(csv_path)
    # Coerce metrics to numeric so None/"None"/empty become NaN and are skipped in means
    for col in ["corr_rmsd", "ligand_rmsd", "atom_type_fraction", "cath_degree"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Add experiment identifier early so success eval can branch
    df['experiment'] = experiment_name

    # Evaluate success for multiple ligand_rmsd thresholds
    ligand_rmsd_thresholds = [1, 2, 4]
    for threshold in ligand_rmsd_thresholds:
        col_name = f'success_ligand_rmsd<{threshold}'
        df[col_name] = df.apply(lambda row: evaluate_success(row, ligand_rmsd_threshold=threshold, criteria=criteria), axis=1)
    df['success_lrmsd_only<4'] = df['ligand_rmsd'] < criteria['ligand_rmsd']
    if symmetric_ligands is not None:
        relaxed = SYMMETRY_RELAXED_LIGAND_RMSD
        symmetric = df['ligand_id'].astype(str).isin(symmetric_ligands)
        success_relaxed = df.apply(lambda row: evaluate_success(row, ligand_rmsd_threshold=relaxed, criteria=criteria), axis=1)
        df['symmetric_ligand'] = symmetric
        df['success_sym_upper_bound<4'] = df['success_ligand_rmsd<4'] | (symmetric & success_relaxed)
        df['success_lrmsd_only_sym_upper_bound<4'] = df['success_lrmsd_only<4'] | (symmetric & (df['ligand_rmsd'] < relaxed))
    
    # Experiment already set above
    
    print(f"{experiment_name}:")
    print(f"  Total samples: {len(df)}")
    for threshold in ligand_rmsd_thresholds:
        col_name = f'success_ligand_rmsd<{threshold}'
        success_count = df[col_name].sum()
        success_rate = df[col_name].mean()
        print(f"  ligand_rmsd < {threshold}: {success_count}/{len(df)} ({success_rate:.2%})")
    print()
    
    return df


def main():
    """Main function to process all experiments and combine results.

    Inputs are read from each directory of ABLATION_SPLIT_DIRS (DATA_ROOT/ablation_dfs/<split>).
    Outputs go to --out_dir/<split> when given, otherwise into the input directory itself.
    Per directory it writes all_experiments_with_success.csv, success_rates.csv,
    success_rates_by_cath_degree.csv and success_rates_lrmsd_only.csv. The last one has, for every
    experiment including the main model, the success rate in the Table 1 columns (different fold =
    cath_degree < 4, same fold = cath_degree == 4, and overall) under the Table 1 criterion and under
    LRMSD-only, plus the symmetry upper bound of both (see load_symmetric_ligands).
    """
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument('--out_dir', default=None,
                    help='write the outputs to <out_dir>/<split> instead of the input directories')
    args = ap.parse_args()
    symmetric_ligands = load_symmetric_ligands(ABLATION_DIRS)
    
    print("=" * 60)
    print("OFFLINE METRICS EVALUATION")
    print("=" * 60)
    print(f"Success Criteria:")
    print(f"  corr_rmsd < {SUCCESS_CRITERIA['corr_rmsd']}")
    print(f"  atom_type_fraction > {SUCCESS_CRITERIA['atom_type_fraction']}")
    print("=" * 60)
    print()
    
    # Process each directory separately
    for ablation_dir in ABLATION_DIRS:
        dir_name = Path(ablation_dir).name
        out_dir = os.path.join(args.out_dir, dir_name) if args.out_dir else ablation_dir
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Processing {dir_name}...")
        print(f"{'='*60}\n")
        
        # Find all CSV files in this directory
        csv_files = []
        excluded_files = {'all_experiments_with_success.csv', 'experiment_summary.csv', 
                          'experiment_summary_easy.csv', 'experiment_summary_hard.csv', 'success_rates.csv',
                          'baseline.csv', 'baseline_sample_1000_without_src_motif.csv', 'baseline_sample_1000_with_src_motif.csv'}
        for csv_path in Path(ablation_dir).glob("*.csv"):
            if csv_path.name not in excluded_files:
                csv_files.append(csv_path)
        
        if not csv_files:
            print(f"No CSV files found in {dir_name}")
            continue
        
        all_experiments = []
        
        # Process each experiment
        for csv_path in sorted(csv_files):
            experiment_name = csv_path.stem  # Filename without extension
            try:
                df_experiment = process_experiment(csv_path, experiment_name, SUCCESS_CRITERIA, symmetric_ligands)
                all_experiments.append(df_experiment)
            except Exception as e:
                print(f"Error processing {experiment_name}: {e}")
                continue
        
        if not all_experiments:
            print(f"No experiments were successfully processed in {dir_name}.")
            continue
        
        # Combine all experiments into one dataframe
        combined_df = pd.concat(all_experiments, ignore_index=True)
        
        # Save combined results
        output_path = os.path.join(out_dir, "all_experiments_with_success.csv")
        combined_df.to_csv(output_path, index=False)
        print("=" * 60)
        print(f"Combined results saved to: {output_path}")
        print(f"Total rows: {len(combined_df)}")
        print("=" * 60)

        # === Baseline visualizations for homology_split ===
        if dir_name == "homology_split":
            baseline_path = os.path.join(ablation_dir, "baseline.csv")
            baseline_df = process_experiment(baseline_path, "baseline", SUCCESS_CRITERIA, symmetric_ligands) \
                if os.path.exists(baseline_path) else pd.DataFrame()
            if not baseline_df.empty and "cath_degree" in baseline_df.columns:
                # Figure 1: counts per cath_degree
                counts = baseline_df.groupby("cath_degree").size()
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(counts.index, counts.values, color="#4C6FFF")
                ax.set_xlabel("CATH degree", fontsize=13)
                ax.set_ylabel("Number of pairs (test set)", fontsize=13)
                ax.set_title("Baseline: pairs per CATH degree (homology split, validation)", fontsize=16)
                ax.tick_params(axis="both", labelsize=11)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.yaxis.grid(True, linestyle="--", alpha=0.4)
                fig.tight_layout()
                fig_path = os.path.join(out_dir, "baseline_pairs_per_cath_degree.png")
                fig.savefig(fig_path, dpi=300)
                plt.close(fig)

                # Figure 2: metric distributions per cath_degree (violin plots)
                fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True)
                metrics_to_plot = [
                    ("Ligand RMSD", "#2E86AB", "ligand_rmsd"),
                    ("Corr RMSD", "#F18F01", "corr_rmsd"),
                    ("Same atom type fraction", "#3DA35D", "atom_type_fraction"),
                ]
                
                for ax, (metric_name, color, col_name) in zip(axes, metrics_to_plot):
                    violin_data = []
                    labels = []
                    for degree in sorted(baseline_df["cath_degree"].unique()):
                        group = baseline_df[baseline_df["cath_degree"] == degree]
                        values = group[col_name].dropna()
                        if len(values) > 0:
                            violin_data.append(values)
                            labels.append(int(degree))
                    
                    parts = ax.violinplot(violin_data, positions=range(len(violin_data)), widths=0.7, 
                                         showmeans=True, showmedians=True)
                    for pc in parts["bodies"]:
                        pc.set_facecolor(color)
                        pc.set_alpha(0.7)
                    
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels)
                    ax.set_title(metric_name, fontsize=18, fontweight="bold")
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
                    if metric_name == "Ligand RMSD":
                        ax.set_ylabel("Value", fontsize=16, fontweight="bold")
                    ax.tick_params(axis="both", labelsize=12)

                fig.text(0.5, 0.02, "CATH degree similarity", ha="center", fontsize=16, fontweight="bold")
                fig.suptitle("Baseline: metric distributions per CATH degree similarity (homology split, validation)", fontsize=18, fontweight="bold")
                fig.tight_layout(rect=[0, 0.05, 1, 0.96])
                fig_path = os.path.join(out_dir, "baseline_success_per_cath_degree.png")
                fig.savefig(fig_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
        
        # Summary by experiment with specific criteria
        summary_data = {}

        # Mean metrics per experiment
        for metric in ["corr_rmsd", "ligand_rmsd", "atom_type_fraction"]:
            if metric in combined_df.columns:
                    summary_data[f"mean_{metric}"] = combined_df.groupby("experiment")[metric].mean().round(5)

        # Split by cath_degree if available
        if "cath_degree" in combined_df.columns:
            # cath_degree < 4 with ligand_rmsd thresholds 1, 2, 4
            easy_df = combined_df[combined_df["cath_degree"] < 4]
            for threshold in [1, 2, 4]:
                col_name = f"success_ligand_rmsd<{threshold}"
                summary_data[f"success_cath<4_rmsd<{threshold}"] = easy_df.groupby("experiment")[col_name].mean().round(5)

            # cath_degree == 4 with ligand_rmsd thresholds 1, 2, 4
            hard_df = combined_df[combined_df["cath_degree"] == 4]
            for threshold in [1, 2, 4]:
                col_name = f"success_ligand_rmsd<{threshold}"
                summary_data[f"success_cath=4_rmsd<{threshold}"] = hard_df.groupby("experiment")[col_name].mean().round(5)

            summary_df = pd.DataFrame(summary_data)

            print("\n" + "=" * 60)
            print(f"Success Rates for {dir_name}:")
            print(summary_df)
            print("=" * 60)

            # Per-exact-cath_degree breakdown: the split above folds
            # cath_degree<4 and cath_degree==4 into two buckets. Report each degree separately
            # (0, 1, 2, 3, 4, ...) instead, so "no difference within 0-3" can actually be checked
            # rather than assumed by the fold.
            #
            # Also include the main model itself (baseline.csv / its two src-motif variants),
            # which the comparison loop above deliberately excludes (it's the reference model,
            # not an ablation/baseline-aligner to compare against). Without this, the exact
            # per-degree numbers behind the abstract's "different fold" success-rate claim
            # wouldn't be in this file at all.
            main_model_frames = []
            for fname in ["baseline.csv", "baseline_sample_1000_with_src_motif.csv",
                          "baseline_sample_1000_without_src_motif.csv"]:
                fpath = os.path.join(ablation_dir, fname)
                if os.path.exists(fpath):
                    mdf = process_experiment(fpath, Path(fname).stem, SUCCESS_CRITERIA, symmetric_ligands)
                    main_model_frames.append(mdf)
            degree_source_df = pd.concat([combined_df] + main_model_frames, ignore_index=True) if main_model_frames else combined_df

            degree_rows = []
            for threshold in [1, 2, 4]:
                col_name = f"success_ligand_rmsd<{threshold}"
                per_degree = degree_source_df.groupby(["experiment", "cath_degree"])[col_name].agg(["mean", "count"])
                for (experiment, degree), row in per_degree.iterrows():
                    degree_rows.append({
                        "experiment": experiment,
                        "cath_degree": degree,
                        "ligand_rmsd_threshold": threshold,
                        "success_rate": round(row["mean"], 5),
                        "n": int(row["count"]),
                    })
            degree_df = pd.DataFrame(degree_rows).sort_values(
                ["experiment", "ligand_rmsd_threshold", "cath_degree"]
            )
            degree_path = os.path.join(out_dir, "success_rates_by_cath_degree.csv")
            degree_df.to_csv(degree_path, index=False)
            print(f"\nPer-exact-degree success rates saved to: {degree_path}")

            fold_rows = []
            for experiment, group in degree_source_df.groupby("experiment"):
                for column, sub in (("different_fold", group[group["cath_degree"] < 4]),
                                    ("same_fold", group[group["cath_degree"] == 4]),
                                    ("overall", group)):
                    row = {
                        "experiment": experiment,
                        "column": column,
                        "n": len(sub),
                        "success_rmsd<4": round(sub["success_ligand_rmsd<4"].mean(), 5),
                        "success_lrmsd_only<4": round(sub["success_lrmsd_only<4"].mean(), 5),
                    }
                    if symmetric_ligands is not None:
                        row["success_sym_upper_bound<4"] = round(sub["success_sym_upper_bound<4"].mean(), 5)
                        row["success_lrmsd_only_sym_upper_bound<4"] = round(sub["success_lrmsd_only_sym_upper_bound<4"].mean(), 5)
                        row["fraction_symmetric_ligand"] = round(sub["symmetric_ligand"].mean(), 5)
                    fold_rows.append(row)
            fold_path = os.path.join(out_dir, "success_rates_lrmsd_only.csv")
            pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
            print(f"LRMSD-only success rates saved to: {fold_path}")
        else:
            print("\nWarning: 'cath_degree' column not found. Cannot compute filtered success rates.")
            summary_df = pd.DataFrame(summary_data)

        # Save summary for this directory
        summary_path = os.path.join(out_dir, "success_rates.csv")
        summary_df.to_csv(summary_path)
        print(f"\nSuccess rates saved to: {summary_path}")


def load_symmetric_ligands(
    ablation_dirs: list[str],
    path: str = SYMMETRY_COUNTS_PATH,
) -> set[str] | None:
    """Ligand codes with more than one RDKit graph automorphism, or None if they cannot be computed.

    The codes are those of baseline.csv in each ablation dir, and the counts come from the JSON
    cache at path. When codes are missing from the cache, scripts/ligand_symmetry.py fetches them
    and rewrites the cache, which needs network access and RDKit. Codes whose fetch or parse failed
    are treated as not symmetric.
    """
    codes = set()
    for ablation_dir in ablation_dirs:
        baseline_path = os.path.join(ablation_dir, "baseline.csv")
        if os.path.exists(baseline_path):
            codes |= set(pd.read_csv(baseline_path)["ligand_id"].astype(str))
    counts = {}
    if os.path.exists(path):
        with open(path) as f:
            counts = json.load(f)
    if codes - set(counts):
        try:
            from ligand_symmetry import load_or_build_symmetry_counts
        except ImportError as e:
            print(f"Cannot build {path} ({e}); skipping the symmetry upper bound.")
            return None
        counts = load_or_build_symmetry_counts(codes, path)
    return {code for code in codes if counts.get(code) is not None and counts[code] > 1}


if __name__ == "__main__":
    main()
