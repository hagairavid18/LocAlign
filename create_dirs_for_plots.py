import pandas as pd
import os
import random
import shutil
from utils.constants import LIGAND_DIR

# Load CSV data
csv_file = "results/baseline_results/2024-07-18_10-26-26_10000.csv"  # Replace with your CSV file path
data = pd.read_csv(csv_file)

# Filter rows with 1 transformation
filtered_data = data[data['n_transformations'] == 1]

# Group by 'cath_degree' and sample 20 pairs for each degree
sampled_data = (
    filtered_data.groupby('cath_degree', group_keys=False)
    .apply(lambda x: x.sample(min(len(x), 20), random_state=42))
)

# Base directory structure
ligands_base_path = LIGAND_DIR  # Adjust if necessary
output_base_path = "example_pairs"  # Replace with the desired output base directory

# Create output directories and copy files
for _, row in sampled_data.iterrows():
    cath_degree = row['cath_degree']
    ligand_id = row['Ligand_ID']
    p1 = row['ref_protein']
    p2 = row['mov_protein']
    
    # Define paths for the new directory based on cath degree
    degree_dir = os.path.join(output_base_path, f"cath_degree_{cath_degree}")
    pair_dir = os.path.join(degree_dir, f"{ligand_id}_{p1}_to_{p2}")
    os.makedirs(pair_dir, exist_ok=True)
    
    # Copy .cif and ligand files
    for protein in [p1, p2]:
        cif_file = os.path.join(ligands_base_path, ligand_id, f"{protein}.cif")
        if os.path.exists(cif_file):
            shutil.copy(cif_file, pair_dir)
        else:
            print(f"Warning: {cif_file} not found")
        
        pdb_ligand_file = os.path.join(ligands_base_path, ligand_id, f"{protein}_ligand.pdb")
        if os.path.exists(pdb_ligand_file):
            shutil.copy(pdb_ligand_file, pair_dir)
        else:
            print(f"Warning: {pdb_ligand_file} not found")
    
    # Define paths for align and pocket files
    ligand_dir = os.path.join(ligands_base_path, ligand_id)
    protein_align_file = os.path.join(ligand_dir, f"{p2}_to_{p1}", "RANSACAlligner_0_protein_0_0.pdb")
    ligand_align_file = os.path.join(ligand_dir, f"{p2}_to_{p1}", "RANSACAlligner_0_ligand_0_0.pdb")
    pocket_file_p1 = os.path.join(ligand_dir, f"{p1}_pocket.pdb")
    pocket_file_p2 = os.path.join(ligand_dir, f"{p2}_pocket.pdb")
    
    # Copy align and pocket files
    for file_path in [protein_align_file, ligand_align_file, pocket_file_p1, pocket_file_p2]:
        if os.path.exists(file_path):
            shutil.copy(file_path, pair_dir)
        else:
            print(f"Warning: {file_path} not found")

print("Directory structure created and files copied.")
