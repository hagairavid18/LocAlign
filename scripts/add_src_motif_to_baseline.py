"""
Script to add src_motif column to baseline.csv by extracting pocket residues
for source proteins using multiprocessing.
"""
import sys
import pandas as pd
from multiprocessing import Pool, cpu_count
import os

sys.path.append('/home/iscb/wolfson/hagairavid/LocAlign')

from miners.objects import Protein


def compute_pocket_for_tuple(args_tuple: tuple[str, str, str], distance_thresh: float = 4.0) -> tuple:
    """
    Multiprocessing-safe wrapper to compute pocket residue IDs aggregating over
    all ligand residues for a given protein/chain/ligand.
    Returns (sorted_pocket_ids_or_None, ligand_residue_count, n_ligand_atoms_list).
    """
    protein_name, chain, ligand = args_tuple
    try:
        protein = Protein(protein_name, chain, ligand, save_models=False)
        ligand_residues = protein.get_ligand_residues()
        residue_count = len(ligand_residues)
        n_ligand_atoms = [len(list(residue.get_atoms())) for residue in ligand_residues]
        
        pockets = set()
        for i in range(residue_count):
            ids = protein.get_pocket_residue_ids(distance_thresh=distance_thresh, ligand_res_idx=i)
            for rid in ids:
                try:
                    pockets.add(int(rid))
                except Exception:
                    pockets.add(rid)
        return (sorted(pockets) if pockets else None), residue_count, n_ligand_atoms
    except Exception as e:
        print(f"Error processing {protein_name} {chain} {ligand}: {e}")
        return None, 0, []


def main():
    # Load baseline.csv
    baseline_path = '/home/iscb/wolfson/hagairavid/LocAlign/ablation_dfs/homology_split/baseline.csv'
    print(f"Loading {baseline_path}...")
    df = pd.read_csv(baseline_path, index_col=0)
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Check if src_motif already exists
    if 'src_motif' in df.columns:
        print("⚠️  src_motif column already exists. Skipping computation.")
        return
    
    # Create args list for source proteins (src_protein, src_chain, src_ligand)
    print(f"\nProcessing {len(df)} source proteins to extract pocket residues (multiprocessing)...")
    args_list = [(row['src_protein'], row['src_chain'], row['src_ligand']) for _, row in df.iterrows()]
    
    src_motif_list = []
    src_ligand_residue_counts = []
    src_n_ligand_atoms_list = []
    
    if args_list:
        processes = min(len(args_list), max(1, cpu_count() - 1))
        print(f"Using {processes} processes...")
        
        with Pool(processes=processes) as pool:
            # Preserve order using imap
            for i, (pockets, count, n_ligand_atoms) in enumerate(pool.imap(compute_pocket_for_tuple, args_list), 1):
                src_motif_list.append(pockets)
                src_ligand_residue_counts.append(count)
                src_n_ligand_atoms_list.append(n_ligand_atoms)
                if i % 50 == 0:
                    print(f"Progress: {i}/{len(args_list)}")
    
    # Add columns to dataframe
    df['src_motif'] = src_motif_list
    df['src_num_ligand_residues'] = src_ligand_residue_counts
    df['src_n_ligand_atoms'] = src_n_ligand_atoms_list
    
    # Save updated baseline.csv
    output_path = baseline_path
    df.to_csv(output_path)
    
    src_multi_res_cases = (df['src_num_ligand_residues'] > 1).sum()
    print(f"\n✅ Completed! Added src_motif, src_num_ligand_residues, src_n_ligand_atoms")
    print(f"Multi-residue source cases: {src_multi_res_cases}")
    print(f"Saved to: {output_path}")
    print(f"Sample row:")
    print(df.iloc[0])


if __name__ == "__main__":
    main()
