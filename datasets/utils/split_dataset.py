import json
import pandas as pd
import argparse
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns

from utils.loading import deserialize_nested_lists

def plot_histogram(data, title, output_dir, filename):
    plt.figure(figsize=(12, 6))
    sns.barplot(x=data.index, y=data.values)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.xlabel('Ligand_ID')
    plt.ylabel('Number of Pairs')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def split_csv(input_csv, output_dir, test_size=0.2, val_size=0.1, group_by_ligand=True):
    # Read the CSV file
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    print(f"Read CSV with {len(df)} rows")
    df = df.drop_duplicates(subset=['ref_protein', 'mov_protein'], keep=False)
    print(f"Removed duplicates, {len(df)} rows remaining")

    for col in  ['bbr', 'bbc']:
        if df[col].apply(lambda x: isinstance(x, str) and x.startswith('[') and x.endswith(']')).any():
            df[col] = df[col].fillna('[]')
            
            df[col] = df[col].apply(lambda x: json.loads(x))
            df[col] = df[col].apply(lambda x: deserialize_nested_lists(x, col))
    
    df['bbr'] = df['bbr'].apply(lambda x: max([max(y) for y in x if len(y) > 0], default=float('-inf')))
    df['bbc'] = df['bbc'].apply(lambda x: max([max(y) for y in x if len(y) > 0], default=float('-inf')))
    df = df[df['bbr'] > 0.3]
    df = df[df['bbc'] > 10]
    print(f"Filtered Best BBR and BBC, {len(df)} rows remaining")
    df = df.groupby('ref_protein').head(50)
    df = df.groupby('mov_protein').head(50)
    print(f"Grouped by ref_protein and mov_protein, {len(df)} rows remaining")
    df = df.groupby('Ligand_ID').head(1000)
    print(f"Grouped by Ligand_ID, {len(df)} rows remaining")

    # Ensure the "Ligand_ID" column exists if group_by_ligand is True
    if group_by_ligand and "Ligand_ID" not in df.columns:
        raise ValueError("The CSV file must contain a 'Ligand_ID' column when group_by_ligand is enabled")

    # Split based on Ligand_ID grouping if specified
    if group_by_ligand:
        # Get unique Ligand_IDs and shuffle
        ligand_ids = df['Ligand_ID'].unique()
        random.seed(42)  # For reproducibility
        random.shuffle(ligand_ids)

        # Calculate number of ligands for each split
        total_ligands = len(ligand_ids)
        test_len = int(total_ligands * test_size)
        val_len = int(total_ligands * val_size)

        # Assign Ligand_IDs to each split
        test_ligands = ligand_ids[:test_len]
        val_ligands = ligand_ids[test_len:test_len + val_len]
        train_ligands = ligand_ids[test_len + val_len:]

        # Create datasets for each split based on Ligand_ID
        train_df = df[df['Ligand_ID'].isin(train_ligands)]
        val_df = df[df['Ligand_ID'].isin(val_ligands)]
        val_df = val_df.groupby('Ligand_ID').head(300)  # Ensure each ligand is present in the validation set
        test_df = df[df['Ligand_ID'].isin(test_ligands)]

        # Plot histograms for each dataset
        plot_histogram(train_df['Ligand_ID'].value_counts(), 'Train Set: Pairs per Ligand', output_dir, 'train_histogram.png')
        plot_histogram(val_df['Ligand_ID'].value_counts(), 'Validation Set: Pairs per Ligand', output_dir, 'val_histogram.png')
        plot_histogram(test_df['Ligand_ID'].value_counts(), 'Test Set: Pairs per Ligand', output_dir, 'test_histogram.png')
        # plot_histogram(train_df['ref_protein'].value_counts(), 'Train Set: ref protein', output_dir, 'train_histogram_ref_protein.png')
        # plot_histogram(val_df['ref_protein'].value_counts(), 'Validation Set: ref protein', output_dir, 'val_histogram_ref_protein.png')
        # plot_histogram(test_df['ref_protein'].value_counts(), 'Test Set: ref protein', output_dir, 'test_histogram.png')

    else:
        # Regular row-wise split without considering Ligand_ID
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Calculate row lengths for each split
        total_rows = len(df)
        test_len = int(total_rows * test_size)
        val_len = int(total_rows * val_size)

        # Split rows into train, validation, and test
        test_df = df.iloc[:test_len]
        val_df = df.iloc[test_len:test_len + val_len]
        train_df = df.iloc[test_len + val_len:]

    # Ensure output directory exists
    

    # Save the splits into CSV files
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"Data split completed. Files and histograms saved in {output_dir}")

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Split CSV by Ligand_ID column or regular row-based split")
    parser.add_argument('--input_csv', type=str, required=True, help="Path to the input CSV file")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output CSV files")
    parser.add_argument('--test_size', type=float, default=0.2, help="Fraction of the data to be used as the test set")
    parser.add_argument('--val_size', type=float, default=0.1, help="Fraction of the total data to be used as validation set")
    parser.add_argument('--group_by_ligand', action='store_true', help="Enable to group rows by Ligand_ID")

    args = parser.parse_args()

    # Call the function to split the CSV
    split_csv(args.input_csv, args.output_dir, test_size=args.test_size, val_size=args.val_size, group_by_ligand=args.group_by_ligand)
