import os
import argparse
import pickle
import numpy as np
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'ScanNet_mini'))
os.environ["KERAS_BACKEND"] = "torch"
import warnings
warnings.simplefilter("ignore")
from ScanNet_mini.predict_features import predict_features
import torch
import gzip
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def _save_scannet_features_worker(args):
    """Worker function for multiprocessing: save a single structure's features."""
    path, features, res_ids, output_dir, list_layers = args
    try:
        if features is None or res_ids is None:
            return None, path, "features could not be generated"
        
        residues_to_atom_indices_idx = list_layers.index('aa_to_atom_indices')
        residues_to_atom_indices = features[residues_to_atom_indices_idx] - features[residues_to_atom_indices_idx][0][0]
        name = os.path.splitext(os.path.basename(path))[0]
        
        sequence_indices_atom = (
            features[list_layers.index('atom_to_aa_indices')] -
            features[list_layers.index('atom_to_aa_indices')][0]
        )[:, 0]
                
        sequence_indices_atom = np.clip(sequence_indices_atom,-1, len(residues_to_atom_indices) -1 )
        residues_to_atom_indices = np.clip(residues_to_atom_indices,-1, len(sequence_indices_atom) -1 )
        
        frames_atom = features[list_layers.index('frames_atom')]
        offset = round(frames_atom[:, 0, :].mean() / 3000) * 3000
        frames_atom[:, 0, :] -= offset
        
        atomic_embeddings = features[list_layers.index('SCAN_filter_activity_atom_1_normalization')]
        residue_embeddings = features[list_layers.index('SCAN_filter_activity_aa_2_normalization')]
        knn_atoms = (
            features[list_layers.index('nearest_neighbor_search_atom')] - 
            features[list_layers.index('nearest_neighbor_search_atom')].min()
        )
        atom_valencies = features[list_layers.index('attributes_atom')][:, 0]
        
        mapping_valency_to_type = np.array([-1, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3])
        atom_types = mapping_valency_to_type[atom_valencies]
        
        chain_name = name.split('_')[0]
        out_path = os.path.join(output_dir, f"{chain_name}_scannet_atoms.pkl")
        os.makedirs(output_dir, exist_ok=True)
        
        data_dict = {
            "sequence_indices_atom": sequence_indices_atom,
            "atomic_embeddings": atomic_embeddings,
            "residue_embeddings": residue_embeddings,
            "residue_ids": res_ids,
            "atomic_frames": frames_atom,
            "aa_to_atom_indices": residues_to_atom_indices,
            "atom_nearest_neighbors": knn_atoms,
            "atom_types": atom_types,
        }
        
        with gzip.open(out_path, "wb") as f:
            pickle.dump(data_dict, f)
        
        return out_path, path, None
    except Exception as e:
        return None, path, str(e)

@torch.inference_mode()
def run_scannet(
    pdb_paths: list[str],
    output_dir: str,
    model: str = 'ScanNet_PPI_noMSA',
    permissive: bool = True
) -> list[str]:
    """
    Run ScanNet feature extraction on a list of PDB/CIF files.
    
    Args:
        pdb_paths: List of PDB or CIF file paths
        output_dir: Directory to store .pkl feature files
        model: ScanNet model to use (default: 'ScanNet_PPI_noMSA')
        permissive: Whether to use permissive mode for predict_features (default: True)
        
    Returns:
        List of output file paths where features were saved
    """
    os.makedirs(output_dir, exist_ok=True)

    list_layers = [
        'attributes_atom',
        'atom_to_aa_indices',  # The atom to amino acid index correspondence
        'aa_to_atom_indices',
        'frames_atom',  # The frames attached to each atom [4,3] matrix
        'nearest_neighbor_search_atom',
        'SCAN_filter_activity_atom_1_normalization',  # Atomic-level embeddings
        'SCAN_filter_activity_aa_2_normalization',  # Amino-acid level embeddings
    ]

    # Predict features
    print("Predicting features...")
    pdb_paths, list_features, list_residue_ids = predict_features(
        pdb_paths,
        layer=list_layers,
        model=model,
        output_format='numpy',
        permissive=permissive
    )

    output_paths = []

    # Prepare save tasks for multiprocessing
    save_tasks = []
    for path, features, res_ids in zip(pdb_paths, list_features, list_residue_ids):
        save_tasks.append((path, features, res_ids, output_dir, list_layers))
    
    # Save with multiprocessing
    print(f"Saving {len(save_tasks)} structures with multiprocessing...")
    num_workers = min(len(save_tasks), max(1, cpu_count() - 1))
    
    if save_tasks:
        with Pool(processes=num_workers) as pool:
            for out_path, path, error in tqdm(pool.imap_unordered(_save_scannet_features_worker, save_tasks), total=len(save_tasks), desc="Saving features"):
                if error is None:
                    output_paths.append(out_path)
                else:
                    print(f"Error processing {path}: {error}")
    
    return output_paths


def extract_scannet(pairs, pdb_dir: str, scannet_dir: str) -> None:
    """
    Run ScanNet feature extraction for proteins described by an iterable of
    Pair-like objects (dataclass `PairHolder`). The function will only process
    structures that don't already have cached features.

    Args:
        pairs: An iterable of objects with attributes 'tar_protein','tar_chain',
               'src_protein','src_chain','ligand' (e.g., a list of PairHolder).
        pdb_dir: Directory containing the non-ligand PDB files (organized by ligand folder).
        scannet_dir: Directory to save ScanNet features
    """
    os.makedirs(scannet_dir, exist_ok=True)
    all_paths = []

    # Expect an iterable of Pair-like objects
    for p in pairs:
        try:
            tar_protein, tar_chain = p.tar_protein, p.tar_chain
            src_protein, src_chain = p.src_protein, p.src_chain
            tar_ligand = getattr(p, 'tar_ligand', None) or p.ligand
            src_ligand = getattr(p, 'src_ligand', None) or p.ligand
        except Exception as e:
            raise ValueError("Each item in 'pairs' must have attributes tar_protein, tar_chain, src_protein, src_chain, ligand") from e

        tar_feature_path = os.path.join(
            scannet_dir,
            f"{tar_protein}{tar_chain}_scannet_atoms.pkl"
        )
        if not os.path.exists(tar_feature_path):
            tar_pdb_path = os.path.join(
                pdb_dir,
                tar_ligand,
                f"{tar_protein}{tar_chain}_non_ligand_.ent"
            )
            if os.path.exists(tar_pdb_path):
                all_paths.append(tar_pdb_path)
            else:
                print(f"❗ Warning: Target PDB path does not exist: {tar_pdb_path}")

        src_feature_path = os.path.join(
            scannet_dir,
            f"{src_protein}{src_chain}_scannet_atoms.pkl"
        )
        if not os.path.exists(src_feature_path):
            src_pdb_path = os.path.join(
                pdb_dir,
                src_ligand,
                f"{src_protein}{src_chain}_non_ligand_.ent"
            )
            if os.path.exists(src_pdb_path):
                all_paths.append(src_pdb_path)
            else:
                print(f"❗ Warning: Source PDB path does not exist: {src_pdb_path}")

    # Resrce duplicates
    print(f"❗ Structures to process with ScanNet: {len(all_paths)}")
    all_paths = list(set(all_paths))
    print(f"❗ Structures to process with ScanNet after deduplication: {len(all_paths)}")

    if len(all_paths) == 0:
        print("❗ No new structures to process. Skipping ScanNet feature extraction.")
        return

    run_scannet(
        pdb_paths=all_paths,
        output_dir=scannet_dir,
        model='ScanNet_PPI_noMSA',
        permissive=True
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run ScanNet feature extraction on PDB/CIF files"
    )
    parser.add_argument(
        "--pdb_paths", 
        nargs="+", 
        required=True, 
        help="List of PDB or CIF files"
    )
    parser.add_argument(
        "--output_dir", 
        required=True, 
        help="Directory to store .pkl feature files"
    )
    parser.add_argument(
        "--model",
        default='ScanNet_PPI_noMSA',
        help="ScanNet model to use (default: ScanNet_PPI_noMSA)"
    )
    parser.add_argument(
        "--permissive",
        action='store_true',
        default=True,
        help="Use permissive mode for predict_features (default: True)"
    )
    args = parser.parse_args()

    # Call run_scannet with command line arguments
    run_scannet(
        pdb_paths=args.pdb_paths,
        output_dir=args.output_dir,
        model=args.model,
        permissive=args.permissive
    )


if __name__ == "__main__":
    main()
