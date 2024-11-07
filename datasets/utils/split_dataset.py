import pandas as pd
import argparse
import os
import random

def split_csv(input_csv, output_dir, test_size=0.2, val_size=0.15):
    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Ensure the "Ligand_ID" column exists
    if "Ligand_ID" not in df.columns:
        raise ValueError("The CSV file must contain a 'Ligand_ID' column")

    # Get the unique Ligand_IDs
    ligand_ids = df['Ligand_ID'].unique()

    # Shuffle ligand IDs for randomness
    random.seed(42)  # For reproducibility
    random.shuffle(ligand_ids)

    # Split Ligand_IDs into train, validation, and test
    total_ligands = len(ligand_ids)
    test_len = int(total_ligands * test_size)
    val_len = int(total_ligands * val_size)

    # Assign Ligand_IDs to each split
    test_ligands = ligand_ids[:test_len]
    val_ligands = ligand_ids[test_len:test_len + val_len]
    train_ligands = ligand_ids[test_len + val_len:]

    # Create datasets for each split
    train_df = df[df['Ligand_ID'].isin(train_ligands)]
    val_df = df[df['Ligand_ID'].isin(val_ligands)]
    test_df = df[df['Ligand_ID'].isin(test_ligands)]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the splits into CSV files
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"Data split completed. Files saved in {output_dir}")

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Split CSV by Ligand_ID column")
    parser.add_argument('--input_csv', type=str, required=True, help="Path to the input CSV file")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output CSV files")
    parser.add_argument('--test_size', type=float, default=0.2, help="Fraction of the data to be used as the test set")
    parser.add_argument('--val_size', type=float, default=0.25, help="Fraction of the test set to be used as validation set")

    args = parser.parse_args()

    # Call the function to split the CSV
    split_csv(args.input_csv, args.output_dir, test_size=args.test_size, val_size=args.val_size)
