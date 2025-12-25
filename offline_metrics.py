import pandas as pd
import os
from pathlib import Path

# Configuration: Success criteria thresholds
SUCCESS_CRITERIA = {
    'corr_rmsd': 2.0,
    'ligand_rmsd': 4.0,
    'atom_type_fraction': 0.5
}

# Directories containing experiment CSVs
ABLATION_DIRS = [
    "/home/iscb/wolfson/hagairavid/LocAlign/ablation_dfs/homology_split",
    "/home/iscb/wolfson/hagairavid/LocAlign/ablation_dfs/ligand_split"
]

def evaluate_success(row, ligand_rmsd_threshold=4.0, criteria=SUCCESS_CRITERIA):
    """
    Evaluate whether a row meets all success criteria with a specific ligand_rmsd threshold.
    
    Args:
        row: DataFrame row with metric columns
        ligand_rmsd_threshold: Custom threshold for ligand_rmsd
        criteria: Dict of metric_name -> threshold
        
    Returns:
        bool: True if all criteria are met
    """
    success = True
    success &= row['corr_rmsd'] < criteria['corr_rmsd']
    success &= row['ligand_rmsd'] < ligand_rmsd_threshold
    success &= row['atom_type_fraction'] > criteria['atom_type_fraction']
    
    return success


def process_experiment(csv_path, experiment_name, criteria=SUCCESS_CRITERIA):
    """
    Load experiment CSV and evaluate success for each row with multiple ligand_rmsd thresholds.
    
    Args:
        csv_path: Path to experiment CSV file
        experiment_name: Name of the experiment
        criteria: Success criteria dict
        
    Returns:
        DataFrame with success columns for different ligand_rmsd thresholds and experiment metadata
    """
    df = pd.read_csv(csv_path)
    
    # Evaluate success for multiple ligand_rmsd thresholds
    ligand_rmsd_thresholds = [1, 2, 4]
    for threshold in ligand_rmsd_thresholds:
        col_name = f'success_ligand_rmsd<{threshold}'
        df[col_name] = df.apply(lambda row: evaluate_success(row, ligand_rmsd_threshold=threshold, criteria=criteria), axis=1)
    
    # Add experiment identifier
    df['experiment'] = experiment_name
    
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
    """Main function to process all experiments and combine results."""
    
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
        print(f"\n{'='*60}")
        print(f"Processing {dir_name}...")
        print(f"{'='*60}\n")
        
        # Find all CSV files in this directory
        csv_files = []
        excluded_files = {'all_experiments_with_success.csv', 'experiment_summary.csv', 
                          'experiment_summary_easy.csv', 'experiment_summary_hard.csv', 'success_rates.csv'}
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
                df_experiment = process_experiment(csv_path, experiment_name, SUCCESS_CRITERIA)
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
        output_path = os.path.join(ablation_dir, "all_experiments_with_success.csv")
        combined_df.to_csv(output_path, index=False)
        print("=" * 60)
        print(f"Combined results saved to: {output_path}")
        print(f"Total rows: {len(combined_df)}")
        print("=" * 60)
        
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
        else:
            print("\nWarning: 'cath_degree' column not found. Cannot compute filtered success rates.")
            summary_df = pd.DataFrame(summary_data)
        
        # Save summary for this directory
        summary_path = os.path.join(ablation_dir, "success_rates.csv")
        summary_df.to_csv(summary_path)
        print(f"\nSuccess rates saved to: {summary_path}")


if __name__ == "__main__":
    main()
