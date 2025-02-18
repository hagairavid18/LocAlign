import sys
import pandas as pd
import os
import shutil
sys.path.append("/home/iscb/wolfson/hagairavid/ligand_alligner")

from utils.constants import LIGAND_DIR
from scripts.chimera_pocket_viz  import process_alignment  # Import the function from the first script

# Load CSV data
csv_file = "results/validation_results/viz-grouping-train-for-viz-corr-offsets-upto2-scalarmatrix-005dp-50epoch-ligandloss-sqrt/Protein_RMSD_Results_31.csv"  # Replace with your CSV file path
data = pd.read_csv(csv_file)

# Filter rows (if needed)
filtered_data = data

# Group by 'CATH Degree' and select top and worst pairs
def select_top_and_worst(group):
    top = group.nsmallest(20, 'Pocket RMSD')  # Smallest RMSD
    worst = group.nlargest(20, 'Pocket RMSD')  # Largest RMSD
    return pd.concat([top, worst])

sampled_data = filtered_data.groupby('CATH Degree', group_keys=False).apply(select_top_and_worst)

# Base directory structure
ligands_base_path = LIGAND_DIR  # Adjust if necessary
output_base_path = "example_pairs_from_validation4"  # Replace with the desired output base directory

# Create output directories and process files
for _, row in sampled_data.iterrows():
    cath_degree = row['CATH Degree']
    ligand_id = row['ligand']
    template = row['tar protein']
    query = row['src protein']
    rmsd = row['Pocket RMSD']
    
    # Define paths for the new directory based on CATH degree
    degree_dir = os.path.join(output_base_path, f"cath_degree_{cath_degree}")
    pair_dir = os.path.join(degree_dir, f"{ligand_id}_{template}_to_{query}")
    os.makedirs(pair_dir, exist_ok=True)
    
    # Call process_alignment for each pair
     # Define paths for align and pocket files
    ligand_dir = os.path.join(ligands_base_path, ligand_id)
    protein_align_file = os.path.join(ligand_dir, f"{query}_to_{template}", "RANSACAlligner_0_protein_0_0.pdb")
    ligand_align_file = os.path.join(ligand_dir, f"{query}_to_{template}", "RANSACAlligner_0_ligand_0_0.pdb")
    pocket_file_p1 = os.path.join(ligand_dir, f"{template}_pocket.pdb")
    pocket_file_p2 = os.path.join(ligand_dir, f"{query}_pocket.pdb")
    
    # Copy align and pocket files
    for file_path in [protein_align_file, ligand_align_file, pocket_file_p1, pocket_file_p2]:
        if os.path.exists(file_path):
            shutil.copy(file_path, pair_dir)
        else:
            print(f"Warning: {file_path} not found")

    # Optionally copy .cif and other files (if still required for other use cases)
    for protein in [template, query]:
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
    try:
        process_alignment(
            base_folder=pair_dir,
            template=template + '_A',
            template_ligand=ligand_id,
            query=query + '_A',
            query_transformation=None  # Update if you have a transformation to pass
        )
    except Exception as e:
        print(f"Error processing alignment for {template} to {query}: {e}")

print("Directory structure created, and alignments processed.")
