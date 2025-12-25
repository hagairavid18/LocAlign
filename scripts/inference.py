import os
import sys
import argparse
import ast
from datetime import datetime

import pandas as pd
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pickle
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'aligner_dl'))

from aligner_dl.datasets import ScanNetDataset
from aligner_dl.models.utils.collate import custom_collate_fn
from aligner_dl.models.utils.misc import build_object
from miners.utils.constants import PairHolder
from miners.objects import Protein
from miners.scripts.chimera_pocket_viz import process_alignment
from scripts.run_scannet import extract_scannet


def _download_non_ligand_worker(args):
    """Top-level worker for multiprocessing (must be picklable)."""
    pdb_name, chain_id, ligand_name, base_save_dir = args
    try:
        Protein(
            pdb_name=pdb_name,
            chain_id=chain_id,
            ligand_name=ligand_name,
            save_models=True,
            ligand_dir=base_save_dir,
        )
        return True, pdb_name, chain_id, ligand_name, None
    except Exception as e:
        return False, pdb_name, chain_id, ligand_name, str(e)


class InferenceRunner:
    """Handles inference pipeline for protein structure alignment."""
    
    DEFAULT_LIGAND = "general"
    
    def __init__(
        self,
        checkpoint_path: str,
        base_save_dir: str,
        csv_path: str | None = None,
        protein_pair: tuple[str, str, str, str] | None = None,
        protein_database_search: tuple[str,str,str] | None = None,
        src_motif: str | None = None,
        tar_motif: str | None = None,
        calibration_model_path: str | None = None,
        max_pLRMSD: float | None = None,
        tar_ligand_id: str | None = None,
        src_ligand_id: str | None = None,
    ) -> None:
        """
        Initialize the inference runner.
        
        Args:
            checkpoint_path (str): Path to model checkpoint
            base_save_dir (str): Base directory to save inference results
            csv_path (str | None): Path to CSV file with protein pairs (must have tar_ligand, src_ligand columns)
            protein_pair (tuple[str, str, str, str] | None): Single protein pair (tar, tar_chain, src, src_chain)
            protein_database_search (tuple[str,str,str] | None): Database search (src_protein, src_chain, database_csv)
            tar_ligand_id (str | None): Ligand for target (pair/database mode, default: 'general')
            src_ligand_id (str | None): Ligand for source (pair/database mode, default: 'general')
            src_motif (str | None): Comma-separated residue IDs for source motif
            tar_motif (str | None): Comma-separated residue IDs for target motif
        """
        self._checkpoint_path = checkpoint_path
        self._tar_ligand_id = tar_ligand_id or self.DEFAULT_LIGAND
        self._src_ligand_id = src_ligand_id or self.DEFAULT_LIGAND
        self._base_save_dir = base_save_dir
        self._csv_path = csv_path
        self._protein_pair = protein_pair
        self._protein_database_search = protein_database_search
        if (calibration_model_path is None):
            default_path = os.path.join(os.path.dirname(checkpoint_path), 'calibration_model.pkl' )
            if os.path.exists(default_path):
                calibration_model_path = default_path
                            
        self._calibration_model_path = calibration_model_path
        self._src_motif = src_motif
        self._tar_motif = tar_motif
        self._max_pLRMSD = max_pLRMSD
        
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._experiment_name = None
        self._checkpoint_dir = None
        self._output_dir = None
        self._model = None
        self._calibration_model = None        
        self._dataloader = None
        
        if self._calibration_model_path is not None:
            try:             
                self._calibration_model = pickle.load( open(self._calibration_model_path,'rb') )
                print('Successfully loaded calibration model')
            except Exception as e:
                print(f'Could not load calibration model, {e}')

    @staticmethod
    def _parse_motif(val):
        """Parse tar_motif field into a list or return None.

        Accepts: None/NaN, already-list, or stringified list (via ast.literal_eval).
        Anything else returns None.
        """
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        if isinstance(val, list):
            return val
        s = str(val).strip()
        if s == "":
            return None
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return None
        return None
    
    def _prepare_dataframe(self) -> None:
        """
        Prepare dataframe from either CSV, protein pair, or database search.
        
        Mode 1 - Pair: protein_pair + optional tar/src_ligand_id (default: 'general')
        Mode 2 - DF: csv_path with tar_ligand and src_ligand columns
        Mode 3 - Database: protein_database_search + src_ligand_id + database CSV with tar_ligand column
        
        Returns:
            None: Sets self._df and self._pairs
        """
        self._pairs: list[PairHolder] = []
        
        if self._protein_pair is not None:
            # Mode 1: Pair mode - use CLI-provided ligands or defaults
            tar, tar_chain, src, src_chain = self._protein_pair
            ph = PairHolder(
                tar_protein=tar, tar_chain=tar_chain, tar_motif=self._tar_motif,
                src_protein=src, src_chain=src_chain, src_motif=self._src_motif,
                tar_ligand=self._tar_ligand_id,
                src_ligand=self._src_ligand_id
            )
            self._pairs.append(ph)
            self._df = pd.DataFrame([ph.to_dict()])
        
        elif self._csv_path is not None:
            df = pd.read_csv(self._csv_path, dtype=str)
            if 'tar_motif' in df.columns:
                df['tar_motif'] = df['tar_motif'].apply(self._parse_motif)
            
            
            if 'ligand' in df.columns:
                df['src_ligand'] = df['ligand']
                df['tar_ligand'] = df['ligand']

                # df = df.rename(columns={'ligand':'tar_ligand'})
            # Validate required columns
            required = ['tar_protein', 'tar_chain', 'src_protein', 'src_chain']
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise ValueError(f"CSV mode requires columns: {missing}. Please add them to your CSV.")
            
            for _, row in df.iterrows():
                ph = PairHolder(
                    tar_protein=row['tar_protein'],
                    tar_chain=row['tar_chain'],
                    tar_motif=row.get('tar_motif', None),
                    src_protein=row['src_protein'],
                    src_chain=row['src_chain'],
                    src_motif=row.get('src_motif', None),
                    tar_ligand=row.get('tar_ligand', self.DEFAULT_LIGAND),
                    src_ligand=row.get('src_ligand', self.DEFAULT_LIGAND),
                )
                self._pairs.append(ph)
            self._df = df
                        
        elif self._protein_database_search is not None:
            # Mode 3: Database search mode - src from CLI, tar from database CSV
            src, src_chain, database_path = self._protein_database_search
            df = pd.read_csv(database_path, dtype=str)

            if 'ligand' in df.columns:
                df['tar_ligand'] = df['ligand']
                df['src_ligand'] = df['ligand']

            if 'tar_motif' in df.columns:
                df['tar_motif'] = df['tar_motif'].apply(self._parse_motif)
            
            # Validate database has required columns
            required = ['tar_protein', 'tar_chain']
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise ValueError(f"Database search mode requires CSV columns: {missing}")
            
            # Add source protein info to all rows
            df['src_protein'] = src
            df['src_chain'] = src_chain
            df['src_motif'] = self._src_motif
            df['src_ligand'] = self._src_ligand_id
        
            for _, row in df.iterrows():
                ph = PairHolder(
                    tar_protein=row['tar_protein'],
                    tar_chain=row['tar_chain'],
                    tar_motif=row.get('tar_motif', None),
                    src_protein=row['src_protein'],
                    src_chain=row['src_chain'],
                    src_motif=row.get('src_motif', None),
                    tar_ligand=row.get('tar_ligand', self.DEFAULT_LIGAND),
                    src_ligand=row['src_ligand'],
                )
                self._pairs.append(ph)
            self._df = df
            
        # Reset index for consistency
        self._df = self._df.reset_index(drop=True)
                                            
    def _setup_directories(self) -> None:
        """
        Setup experiment and output directories.
        
        Returns:
            None: Sets self._experiment_name, self._checkpoint_dir, and self._output_dir
        
        Raises:
            FileNotFoundError: If checkpoint directory doesn't exist
        """
        self._experiment_name = self._checkpoint_path.split('/')[-2]
        self._checkpoint_dir = self._checkpoint_path.rsplit('/', 1)[0]
        
        if not os.path.isdir(self._checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {self._checkpoint_dir}")
        
        # Determine output directory (always use experiment name + timestamp)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._output_dir = os.path.join(self._base_save_dir, f"{self._experiment_name}_{timestamp}")
        if self._protein_database_search is not None:
            protein, chain, _ = self._protein_database_search
            self._output_dir = os.path.join(self._base_save_dir, f"database_search_{protein}_{chain}_{timestamp}")
        
        os.makedirs(self._output_dir, exist_ok=True)
        print(f"Saving results to: {self._output_dir}")
        # create a local cache inside the base save dir for embeddings, pdbs and fastas
        cache_root = os.path.join(self._base_save_dir, '.cache')
        self._cache_paths = {
            'esm_embeddings': os.path.join(cache_root, 'esm_embeddings'),
            'scannet_embeddings': os.path.join(cache_root, 'scannet_embeddings'),
            'fasta_files': os.path.join(cache_root, 'fasta_files'),
            'pdb_files': os.path.join(cache_root, 'pdb_files'),
        }
        for p in self._cache_paths.values():
            os.makedirs(p, exist_ok=True)
        print(f"Created local cache at: {cache_root}")
    
    def _save_non_ligand_models(self) -> None:
        """
        Save non-ligand PDB models for all proteins in the dataframe.
        Only saves unique protein-chain-ligand combinations to avoid duplicates.
        
        Returns:
            None: Downloads and saves PDB files to self._base_save_dir
        """
        # Collect unique protein-chain-ligand combinations from PairHolder list
        unique_combinations = set()
        for ph in self._pairs:
            unique_combinations.add((ph.tar_protein, ph.tar_chain, ph.tar_ligand))
            unique_combinations.add((ph.src_protein, ph.src_chain, ph.src_ligand))

        # avoid redownloading cached structures
        unique_combinations = {combo for combo in unique_combinations if not os.path.exists(os.path.join(self._cache_paths['pdb_files'], combo[2], f"{combo[0]}{combo[1]}_non_ligand_.ent"))}

        # Save models with multiprocessing and progress bar
        combos = list(unique_combinations)
        total = len(combos)
        print(f"Saving {total} unique non-ligand models (parallel)...")

        # Save PDBs into the local cache pdb_files folder to avoid polluting output dir
        args_list = [(p, c, l, self._cache_paths['pdb_files']) for (p, c, l) in combos]
        processes = min(total, max(1, cpu_count() - 1))
        # collect failed combos with error messages
        failed_combos: list[tuple[str, str, str, str]] = []
        if combos:
            with Pool(processes=processes) as pool:
                for success, pdb_name, chain_id, ligand_name, err in tqdm(pool.imap_unordered(_download_non_ligand_worker, args_list), total=total, desc="Downloading PDB files"):
                    if not success:
                        print(f"Failed to download {pdb_name} chain {chain_id} ligand {ligand_name}: {err}")
                        failed_combos.append((pdb_name, chain_id, ligand_name, err))

        if failed_combos:
            failed_set = set((p, c, l) for p, c, l, _ in failed_combos)
            err_map = {(p, c, l): err for p, c, l, err in failed_combos}

            # mark PairHolder.message and failure_message for any pair referencing a failed combo
            for ph in self._pairs:
                tar_combo = (ph.tar_protein, ph.tar_chain, ph.tar_ligand)
                src_combo = (ph.src_protein, ph.src_chain, ph.src_ligand)
                msgs = []
                if tar_combo in failed_set:
                    msgs.append(f"tar_missing:{err_map.get(tar_combo)}")
                if src_combo in failed_set:
                    msgs.append(f"src_missing:{err_map.get(src_combo)}")
                if msgs:
                    failure_msg = ';'.join(msgs)
                    ph.message = failure_msg
                    ph.failure_message = failure_msg

            # Preserve all pairs (including failed) for final results
            self._all_pairs = list(self._pairs)
            # Build kept list for feature extraction and inference
            kept = [p for p in self._pairs if p.message is None]

            self._pairs = kept

            if not kept:
                raise RuntimeError("All protein pairs were resrced because required non-ligand models failed to download. Aborting inference.")
    
    def _extract_features(self) -> None:
        """
        Run ScanNet feature extraction.
        
        Returns:
            None: Generates feature files in the local ScanNet cache
        """
        print("Running ScanNet feature extraction into local cache...")

        extract_scannet(self._pairs, self._cache_paths['pdb_files'], self._cache_paths['scannet_embeddings'])

    def _write_resrced_pairs(self) -> None:
        """
        Persist resrced pairs (those with non-empty message) to the output directory.
        This is called after preprocessing and before model/dataloader creation.
        """
        resrced = [p for p in getattr(self, '_pairs', []) if p.message is not None]
        if not resrced:
            return

        # Include the message in the saved CSV for debugging
        rows = []
        for p in resrced:
            d = p.to_dict()
            d['message'] = p.message
            rows.append(d)

        resrced_df = pd.DataFrame(rows)
        resrced_path = os.path.join(self._output_dir, 'resrced_pairs_due_to_download_failures.csv')
        resrced_df.to_csv(resrced_path, index=False)
        print(f"Wrote {len(resrced_df)} resrced pairs to: {resrced_path}")
    
    def _prepare_dataloader(self) -> None:
        """
        Build dataset and dataloader.
        
        Returns:
            None: Sets self._dataloader
        
        Raises:
            FileNotFoundError: If dataset config file doesn't exist
        """
        print("Preparing dataset and dataloader...")
        dataset_config_path = os.path.join(self._checkpoint_dir, "dataset_config.yaml")
        if not os.path.isfile(dataset_config_path):
            raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")
        
        with open(dataset_config_path) as f:
            dataset_config = yaml.safe_load(f)

        # Build and persist the filtered dataframe from PairHolder objects here
        kept_pairs = [p for p in getattr(self, '_pairs', []) if p.message is None]
        if not kept_pairs:
            raise RuntimeError("No valid protein pairs remain to build the dataset.")

        filtered_df = pd.DataFrame([p.to_dict() for p in kept_pairs])
        # Add stable pair_idx to ensure correct mapping during result processing
        filtered_df['pair_idx'] = list(range(len(kept_pairs)))
        filtered_csv = os.path.join(self._output_dir, 'filtered_pairs.csv')
        filtered_df.to_csv(filtered_csv, index=False)
        self._df = filtered_df

        # Update dataset config for inference
        dataset_config['args']['df_path'] = filtered_csv
        # point the dataset to the local cached pdb files
        dataset_config['args']['base_data_path'] = self._cache_paths['pdb_files']
        # point dataset to the local cache for scannet and esm embeddings
        dataset_config['args']['base_scannet_path'] = self._cache_paths['scannet_embeddings']
        dataset_config['args']['base_esm_embedding_path'] = self._cache_paths['esm_embeddings']
        dataset_config['args']['inference'] = True
        dataset_config['args']['ligand_column'] = 'ligand'
        dataset_config['args']['tar_ligand_column'] = 'tar_ligand'
        dataset_config['args']['src_ligand_column'] = 'src_ligand'
        
        dataset = ScanNetDataset(**dataset_config['args'])
        self._dataloader = DataLoader(
            dataset,
            batch_size=8,
            num_workers=0,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            shuffle=False
        )
    
    def _load_model(self) -> None:
        """
        Build model and load checkpoint.
        
        Returns:
            None: Sets self._model and srces it to device
        
        Raises:
            FileNotFoundError: If model config file doesn't exist
        """
        print("Building model from config...")
        model_config_path = os.path.join(self._checkpoint_dir, "model_config.yaml")
        if not os.path.isfile(model_config_path):
            raise FileNotFoundError(f"Model config not found: {model_config_path}")
       
        with open(model_config_path) as f:
            model_config = yaml.safe_load(f)
        
        self._model = build_object(model_config, 'models')
        
        print("Loading model checkpoint...")
        checkpoint = torch.load(self._checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        self._model.load_state_dict(state_dict)

        self._model.to(self._device)
        self._model.eval()
    
    def _run_inference(self) -> None:
        """
        Run inference on all batches.
        
        Returns:
            None: Processes all batches and saves results to self._output_dir
        """
        print("Running inference...")
        all_outputs = []
        with torch.no_grad():
            pbar = tqdm(self._dataloader, total=len(self._dataloader), desc="Running inference")
            for idx, batch in enumerate(pbar):
                curr_outputs = self._model.inference_step(batch)
                all_outputs.append(curr_outputs)
        
        for idx, preds in enumerate(tqdm(all_outputs, total=len(all_outputs), desc="Processing and saving results")):
            batch_size = len(preds['metadata'])
            for b in range(batch_size):
                pair_idx = preds['metadata'][b]['pair_idx']
                ph = self._pairs[pair_idx]
                self._process_visualization(ph, preds, b)
        self._save_results()
        print(f"✅ All predictions processed and visualized, saved to {self._output_dir}")
    
    def _process_visualization(self, ph: PairHolder, preds: dict, batch_idx: int = 0) -> int:
        """
        Process a single sample from a batch and save results.
        
        Args:
            ph (PairHolder): The pair holder for this sample
            preds (dict): Pre-computed predictions from model.inference_step() for entire batch
            batch_idx (int): Index within the batch to process
            
        Returns:
            int: Number of correspondences
        """

        # Use pre-computed predictions instead of running inference again
        metadata = preds["metadata"][batch_idx]

        # Filter correspondences above threshold (25% of max correlation)
        corr_vals = preds["corr_values"][batch_idx]
        
        sorted_corr_vals = torch.sort(corr_vals,0,descending=True)
        cumulative_corr_vals = torch.cumsum(sorted_corr_vals.values,0)
        above_threshold_indices = sorted_corr_vals.indices[cumulative_corr_vals <= 0.66]
        
        # Ensure at least 3 correspondences
        if len(above_threshold_indices) < 3:
            above_threshold_indices = torch.topk(corr_vals, 3)[1]

        # take up to 20 correspondences
        # if len(above_threshold_indices) > 20:
        #     above_threshold_indices = above_threshold_indices[:20]

        num_correspondences = int(above_threshold_indices.numel())

        top_corr_values = preds["corr_values"][batch_idx][above_threshold_indices]
        top_corr_indices = preds["corr_indices"][batch_idx][above_threshold_indices]
        top_corr_indices_atom = preds["corr_atom_indices"][batch_idx][above_threshold_indices]

        # Prepare save paths and folder name
        tar = metadata["tar_protein"]
        src = metadata["src_protein"]
        tar_chain = metadata["tar_chain"]
        src_chain = metadata["src_chain"]
        tar_ligand = metadata.get("tar_ligand", metadata.get("ligand", "general"))
        src_ligand = metadata.get("src_ligand", metadata.get("ligand", "general"))
        ligand_tag = tar_ligand if tar_ligand == src_ligand else f"{tar_ligand}_{src_ligand}"

        corr_rmsd = preds['loss_dict']['per_sample']['corr_rmsd'][batch_idx].item()
        gap = preds['loss_dict']['per_sample']['gap'][batch_idx].item()
        emb = preds['loss_dict']['per_sample']['embedding'][batch_idx].item()
        radius = preds['loss_dict']['per_sample']['radius'][batch_idx].item()
        
        # Transformation of variables for interpretability
        perplexity = int( np.exp(gap * np.log(400)) )
        attribute_similarity = emb / (gap * np.log(400) )
        radius_gyration = radius * ( 1.3 * perplexity ** (0.4) )
        if self._calibration_model is not None:
            features = np.array([attribute_similarity,corr_rmsd,radius_gyration,perplexity])[None] # ['normalized_embedding_similarity','corr_rmsd','radius_of_gyration','perplexity']
            pLRMSD = self._calibration_model.predict(features)[0]
        else:
            pLRMSD = (  2 *  (1 - emb / np.log(400) ) + 2 * corr_rmsd + 1 * radius_gyration ) # A dummy formula.
            
        # Store metrics on the PairHolder object (explicit fields)
        ph.pLRMSD = pLRMSD
        ph.perplexity = perplexity
        ph.attribute_similarity = attribute_similarity
        ph.correspondence_rmsd = corr_rmsd
        ph.radius_gyration = radius_gyration
        ph._embedding = emb
        ph._gap = gap
        ph._corr_rmsd = corr_rmsd
        ph._radius = radius
        
        if (self._protein_database_search is not None ) & (self._max_pLRMSD is not None): 
            if pLRMSD>= self._max_pLRMSD: # Skip building output file in this case.
                return num_correspondences
                
        save_folder = os.path.join(
            self._output_dir,
            f"{tar.lower()}{tar_chain}_{src.lower()}{src_chain}_{ligand_tag}_"
            f"pLRMSD{pLRMSD:.2f}_perp{perplexity:03d}_attr{attribute_similarity:.2f}_corr{corr_rmsd:.2f}_rad{radius:.2f}"
        )


        os.makedirs(save_folder, exist_ok=True)

        trans_dict = preds["transformation_dict"]

        # Convert to numpy arrays without saving to file
        R_np = trans_dict['pred_R'][batch_idx].detach().cpu().numpy()
        t_np = trans_dict['pred_t'][batch_idx].detach().cpu().numpy()
        # Save transformation (rotation and translation) for this alignment
        try:
            transform_path = os.path.join(save_folder, 'transformation.npz')
            np.savez_compressed(transform_path, R=R_np, t=t_np)
        except Exception as e:
            print(f"Failed to save transformation: {e}")
        
        # Generate visualization using Chimera via process_alignment (pass arrays directly)
        try:
            process_alignment(
                base_folder=save_folder,
                cache_dir= os.path.join(self._base_save_dir, '.cache'),
                template=tar + metadata['tar_chain'],
                template_ligand=tar_ligand,
                query=src + metadata['src_chain'],
                query_ligand=src_ligand,
                query_transformation=(R_np, t_np),
                corr_values=top_corr_values.detach().cpu().numpy(),
                corr_indices=top_corr_indices.detach().cpu().numpy(),
                atom_indexes_list=top_corr_indices_atom.detach().cpu().numpy()
            )
        except Exception as e:
            print(f"Unexpected error during visualization: {e}")

        # Record output folder on the pair for traceability
        ph.output_folder = save_folder

        return num_correspondences
    
    def _save_results(self) -> None:
        # Combine succeeded pairs (with metrics) and failed pairs (with NaN metrics)
        all_pairs = getattr(self, '_all_pairs', self._pairs)
        rows = [p.to_dict() for p in all_pairs]
        results_df = pd.DataFrame(rows)
        # Sort by pLRMSD, pushing NaNs (failed pairs) to the bottom
        results_df = results_df.sort_values(by='pLRMSD', ascending=True, na_position='last')
        results_df.to_csv(os.path.join(self._output_dir, "inference_results.csv"), index=False, float_format='%.3f')
    
    def run(self) -> None:
        """
        Execute the complete inference pipeline.
        
        Returns:
            None: Runs full pipeline and saves all results
        """
        self._prepare_dataframe()
        self._setup_directories()
        
        # Always download PDB models (we have per-chain ligands now)
        self._save_non_ligand_models()

        self._extract_features()
        # persist resrced pairs (with messages) now, before creating the dataloader/model
        self._write_resrced_pairs()
        self._prepare_dataloader()
        self._load_model()
        self._run_inference()
        self._save_results()


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using a trained model.")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/baseline-embed01-ligand5-corr1-recycle4-radius05-sched-corr2/epoch=9-step=87120.ckpt",
        help="Path to model checkpoint (default: preconfigured baseline)."
    )
    parser.add_argument(
        "--tar_ligand_id",
        type=str,
        default=None,
        help="Ligand for target protein. Used in --protein_pair and --protein_database_search modes (default: 'general'). Ignored in --csv_path mode."
    )
    parser.add_argument(
        "--src_ligand_id",
        type=str,
        default=None,
        help="Ligand for source protein. Used in --protein_pair and --protein_database_search modes (default: 'general'). Ignored in --csv_path mode."
    )
    parser.add_argument(
        "--base_save_dir",
        type=str,
        default="inference_results",
        help="Base directory to save inference results."
    )
    # --save_dir resrced: output directory is always created under base_save_dir with a timestamp

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--csv_path",
        type=str,
        help="DF Mode: CSV with columns [tar_protein, tar_chain, src_protein, src_chain]. Optional: tar_ligand, src_ligand (default: 'general')."
    )
    group.add_argument(
        "--protein_pair",
        nargs=4,
        metavar=("TAR_PROTEIN", "TAR_CHAIN", "SRC_PROTEIN", "SRC_CHAIN"),
        help="Pair Mode: Single protein pair. Use --tar_ligand_id and --src_ligand_id to specify ligands (default: 'general')."
    )
    group.add_argument(
        '--protein_database_search',
        nargs=3,
        metavar=("SRC_PROTEIN","SRC_CHAIN","DATABASE_CSV"),
        help="Database Mode: Search source protein against database. CSV must have [tar_protein, tar_chain]. Optional: tar_ligand (default: 'general'). Use --src_ligand_id for source."
    )
    
    parser.add_argument(
        "--src_motif",
        type=str,
        default=None,
        help="Comma-separated residue IDs for source motif (e.g., '10,11,12'). Only used with --protein_pair."
    )
    parser.add_argument(
        "--tar_motif",
        type=str,
        default=None,
        help="Comma-separated residue IDs for target motif (e.g., '20,21,22'). Only used with --protein_pair."
    )
    parser.add_argument(
        "--max_pLRMSD",
        type=float,
        default = 4.,
        help = 'In protein_database_search mode, do not generate output directory if pLRMSD is above this threshold'
    )
    
    parser.add_argument(
        "--calibration_model_path",
        type=str,
        default=None,
        help="Path to calibration model (pickle file) for pLRMSD prediction."
    )
    return parser.parse_args()

@torch.inference_mode()
def main():
    """Main entry point for the inference script."""
    args = parse_args()
    
    # Initialize inference runner
    runner = InferenceRunner(
        checkpoint_path=args.checkpoint,
        calibration_model_path=args.calibration_model_path,
        tar_ligand_id=args.tar_ligand_id,
        src_ligand_id=args.src_ligand_id,
        base_save_dir=args.base_save_dir,
        csv_path=args.csv_path,
        protein_pair=args.protein_pair,
        protein_database_search=args.protein_database_search,
        max_pLRMSD=args.max_pLRMSD,
        src_motif=args.src_motif,
        tar_motif=args.tar_motif
    )
    
    # Run the complete inference pipeline
    runner.run()


if __name__ == "__main__":
    main()
