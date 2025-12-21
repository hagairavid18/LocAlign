import os
import sys
import argparse
from datetime import datetime
from typing import Any

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
    
    def __init__(
        self,
        checkpoint_path: str,
        ligand_id: str,
        base_save_dir: str,
        csv_path: str | None = None,
        protein_pair: tuple[str, str, str, str] | None = None,
        protein_database_search: tuple[str,str,str] | None = None,
        src_motif: str | None = None,
        tar_motif: str | None = None,
        calibration_model_path: str | None = None,
        max_pLRMSD: float | None = None,
    ) -> None:
        """
        Initialize the inference runner.
        
        Args:
            checkpoint_path (str): Path to model checkpoint
            ligand_id (str): Ligand PDB name
            base_save_dir (str): Base directory to save inference results
            csv_path (str | None): Path to CSV file with protein pairs (optional)
            protein_pair (tuple[str, str, str, str] | None): Single protein pair as 
                (tar, tar_chain, src, src_chain) (optional)
            src_motif (str | None): Comma-separated residue IDs for source motif (optional)
            tar_motif (str | None): Comma-separated residue IDs for target motif (optional)
        """
        self._checkpoint_path = checkpoint_path
        self._ligand_id = ligand_id
        self._base_save_dir = base_save_dir
        self._csv_path = csv_path
        self._protein_pair = protein_pair
        self._protein_database_search = protein_database_search
        self._calibration_model_path = calibration_model_path
        self._src_motif = src_motif
        self._tar_motif = tar_motif
        self._max_pLRMSD = max_pLRMSD
        
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._df = None
        self._csv_output_path = None
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
    
    def _prepare_dataframe(self) -> None:
        """
        Prepare dataframe from either CSV or protein pair.
        
        Returns:
            None: Sets self._df and self._csv_output_path
        """
        # Build PairHolder list from CSV or single pair input.
        self._pairs: list[PairHolder] = []
        if self._protein_pair is not None:
            tar, tar_chain, src, src_chain = self._protein_pair
            ph = PairHolder(tar_protein=tar, tar_chain=tar_chain, tar_motif=self._tar_motif,
                            src_protein=src, src_chain=src_chain, src_motif=self._src_motif,
                            ligand=self._ligand_id)
            self._pairs.append(ph)
            # create initial dataframe and csv path (temporary)
            df_dict = ph.to_dict()
            # if self._src_motif:
            #     df_dict['src_motif'] = self._src_motif
            # if self._tar_motif:
            #     df_dict['tar_motif'] = self._tar_motif
            self._df = pd.DataFrame([df_dict])
            # self._csv_output_path = "temp_csv.csv"
        elif self._csv_path is not None:
            raw_df = pd.read_csv(self._csv_path,dtype=str)
            # Rename columns to match expected format
            for _, row in raw_df.iterrows():
                ph = PairHolder(
                    tar_protein=row['tar_protein'],
                    tar_chain=row['tar_chain'],
                    tar_motif=row.get('tar_motif', None),
                    src_protein=row['src_protein'],
                    src_chain=row['src_chain'],
                    src_motif=row.get('src_motif', None),
                    ligand=row.get('ligand', self._ligand_id),
                )
                self._pairs.append(ph)
            self._df = raw_df
            # self._csv_output_path = self._csv_path                
        elif self._protein_database_search is not None:
            src,src_chain,protein_template_database_path = self._protein_database_search
            src_motif = self._src_motif
            df = pd.read_csv(protein_template_database_path,dtype=str)
            df['src_protein'] = src
            df['src_chain'] = src_chain
            df['src_motif'] = src_motif
            del df['ligand'] # For now... Only to avoid crashing                        
            for _, row in df.iterrows():
                ph = PairHolder(
                    tar_protein=row['tar_protein'],
                    tar_chain=row['tar_chain'],
                    tar_motif=row.get('tar_motif', None),
                    src_protein=row['src_protein'],
                    src_chain=row['src_chain'],
                    src_motif=row.get('src_motif', None),
                    ligand=row.get('ligand', self._ligand_id),
                )
                self._pairs.append(ph)            
            self._df = df
            
        self._csv_output_path = "inference_results.csv"
                        
        self._results_df = self._df.copy()
        for metric in ['pLRMSD','perplexity','attribute_similarity','correspondence_rmsd','radius_gyration',
                        '_embedding','_gap','_corr_rmsd','_radius']:
            self._results_df[metric] = np.nan
        self._results_df['index'] = self._results_df['tar_protein'] + '_' +\
                                            self._results_df['tar_chain'] + '_' +\
                                            self._results_df['tar_motif'].map(lambda x: 'none' if ( isinstance(x,float) or  (x in [None,'nan','None'] ) ) else str(x) ) + '_' +\
                                            self._results_df['src_protein'] + '_' +\
                                            self._results_df['src_chain'] + '_' +\
                                            self._results_df['src_motif'].map(lambda x: 'none' if (isinstance(x,float) or  (x in [None,'nan','None']) ) else str(x) )
                                            
                        
            
    
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
            unique_combinations.add((ph.tar_protein, ph.tar_chain, ph.ligand))
            unique_combinations.add((ph.src_protein, ph.src_chain, ph.ligand))

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

            # mark PairHolder.message for any pair tarerencing a failed combo
            for ph in self._pairs:
                tar_combo = (ph.tar_protein, ph.tar_chain, ph.ligand)
                src_combo = (ph.src_protein, ph.src_chain, ph.ligand)
                msgs = []
                if tar_combo in failed_set:
                    msgs.append(f"tar_missing:{err_map.get(tar_combo)}")
                if src_combo in failed_set:
                    msgs.append(f"src_missing:{err_map.get(src_combo)}")
                if msgs:
                    ph.message = ';'.join(msgs)

            # Build resrced and filtered dataframes from PairHolder list
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
        filtered_csv = os.path.join(self._output_dir, 'filtered_pairs.csv')
        filtered_df.to_csv(filtered_csv, index=False)
        self._df = filtered_df
        # self._csv_output_path = filtered_csv

        # Update dataset config for inference
        dataset_config['args']['df_path'] = filtered_csv
        # point the dataset to the local cached pdb files
        dataset_config['args']['base_data_path'] = self._cache_paths['pdb_files']
        # point dataset to the local cache for scannet and esm embeddings
        dataset_config['args']['base_scannet_path'] = self._cache_paths['scannet_embeddings']
        dataset_config['args']['base_esm_embedding_path'] = self._cache_paths['esm_embeddings']
        dataset_config['args']['inference'] = True
        dataset_config['args']['ligand_column'] = 'ligand'
        
        dataset = ScanNetDataset(**dataset_config['args'])
        self._dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=custom_collate_fn,
            pin_memory=True
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
        with torch.no_grad():
            pbar = tqdm(self._dataloader, total=len(self._dataloader), desc="Running inference")
            for idx, batch in enumerate(pbar):
                num_corr = self._process_batch(idx, batch)
                # update tqdm description with last batch correspondence count
                try:
                    pbar.set_description(f"Running inference | n_corr={num_corr}")
                except Exception:
                    pass

        print(f"✅ All predictions processed and visualized, saved to {self._output_dir}")
    
    def _process_batch(self, idx: int, batch: Any) -> int:
        """
        Process a single batch and save results.
        
        Args:
            idx (int): Batch index
            batch (Any): Batch dictionary containing protein pair data
            
        Returns:
            None: Saves predictions and visualizations to self._output_dir
        """
        # srce tensors to device (model also handles this internally, but we keep it explicit here)
        batch = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Run model inference for this batch (batch_size is 1)
        preds = self._model.inference_step(batch)
        metadata = preds["metadata"][0]

        # Filter correspondences above threshold (25% of max correlation)
        corr_vals = preds["corr_values"][0]
        
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

        top_corr_values = preds["corr_values"][0][above_threshold_indices]
        top_corr_indices = preds["corr_indices"][0][above_threshold_indices]
        top_corr_indices_atom = preds["corr_atom_indices"][0][above_threshold_indices]

        # Prepare save paths and folder name
        tar,tar_chain = metadata["tar_protein"], metadata["tar_chain"]
        src,src_chain = metadata["src_protein"], metadata["src_chain"]
        ligand = metadata.get("ligand", "general")

        corr_rmsd = preds['loss_dict']['corr_rmsd'].item()
        gap = preds['loss_dict']['gap'].item()
        emb = preds['loss_dict']['embedding'].item()
        radius = preds['loss_dict']['radius'].item()
        
        # Transformation of variables for interpretability
        perplexity = int( np.exp(gap * np.log(400)) )
        attribute_similarity = emb / (gap * np.log(400) )
        radius_gyration = radius * ( 1.3 * perplexity ** (0.4) )
        if self._calibration_model is not None:
            features = np.array([attribute_similarity,corr_rmsd,radius_gyration,perplexity])[None] # ['normalized_embedding_similarity','corr_rmsd','radius_of_gyration','perplexity']
            pLRMSD = self._calibration_model.predict(features)[0]
        else:
            pLRMSD = (  2 *  (1 - emb / np.log(400) ) + 2 * corr_rmsd + 1 * radius_gyration ) # A dummy formula.
            
        clean_motif_name = lambda x: 'none' if ( isinstance(x,float) or  (x in [None,'nan','None'] ) ) else str(x)
        index = metadata["tar_protein"] + '_' +\
                metadata['tar_chain'] + '_' +\
                clean_motif_name(metadata['tar_motif']) + '_' +\
                metadata['src_protein'] + '_' +\
                metadata['src_chain'] + '_' +\
                clean_motif_name(metadata['src_motif'])
        bool_index = (self._results_df['index']==index)
        self._results_df['pLRMSD'][bool_index] = pLRMSD
        self._results_df['perplexity'][bool_index] = perplexity
        self._results_df['attribute_similarity'][bool_index] = attribute_similarity
        self._results_df['correspondence_rmsd'][bool_index] = corr_rmsd
        self._results_df['radius_gyration'][bool_index] = radius_gyration                
        self._results_df['_embedding'][bool_index] = emb
        self._results_df['_gap'][bool_index] = gap        
        self._results_df['_corr_rmsd'][bool_index] = corr_rmsd
        self._results_df['_radius'][bool_index] = radius
        
        if (self._protein_database_search is not None ) & (self._max_pLRMSD is not None): 
            if pLRMSD>= self._max_pLRMSD: # Skip building output file in this case.
                return num_correspondences
                
        save_folder = os.path.join(
            self._output_dir,
            f"{tar.lower()}{tar_chain}_{src.lower()}{src_chain}_{ligand}_"
            f"pLRMSD{pLRMSD:.2f}_perp{perplexity:03d}_attr{attribute_similarity:.2f}_corr{corr_rmsd:.2f}_rad{radius:.2f}"
        )

        # save_folder = os.path.join(
        #     self._output_dir,
        #     f"{tar}_{src}_{ligand}_"
        #     f"corr{corr_rmsd:.2f}_gap{gap:.2f}_emb{emb:.2f}_radius{radius:.2f}"
        # )
        os.makedirs(save_folder, exist_ok=True)

        trans_dict = preds["transformation_dict"]

        # Convert to numpy arrays without saving to file
        R_np = trans_dict['pred_R'][0].detach().cpu().numpy()
        t_np = trans_dict['pred_t'][0].detach().cpu().numpy()
        # Save transformation (rotation and translation) for this alignment
        try:
            transform_path = os.path.join(save_folder, 'transformation.npz')
            np.savez_compressed(transform_path, R=R_np, t=t_np)
        except Exception as e:
            print(f"Failed to save transformation for batch {idx}: {e}")
        
        # Generate visualization using Chimera via process_alignment (pass arrays directly)
        try:
            process_alignment(
                base_folder=save_folder,
                ligand=ligand,
                cache_dir= os.path.join(self._base_save_dir, '.cache'),
                template=tar + metadata['tar_chain'],
                query=src + metadata['src_chain'],
                query_transformation=(R_np, t_np),
                corr_values=top_corr_values.detach().cpu().numpy(),
                corr_indices=top_corr_indices.detach().cpu().numpy(),
                atom_indexes_list=top_corr_indices_atom.detach().cpu().numpy()
            )
        except Exception as e:
            print(f"Unexpected error during visualization: {e}")

        return num_correspondences
    
    def _save_results(self) -> None:
        del self._results_df['index']        
        self._results_df = self._results_df.sort_values(by='pLRMSD',ascending=True)
        self._results_df['perplexity'] = self._results_df['perplexity']
        self._results_df.to_csv( os.path.join(self._output_dir,"inference_results.csv"),index=False,float_format='%.3f')
    
    def run(self) -> None:
        """
        Execute the complete inference pipeline.
        
        Returns:
            None: Runs full pipeline and saves all results
        """
        self._prepare_dataframe()
        self._setup_directories()
        
        if self._ligand_id:
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
        "--ligand_id",
        type=str,
        default="general",
        help="The ligand PDB name, if both inputs bind the same one."
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
        help="Path to CSV file with tar/src chains."
    )
    group.add_argument(
        "--protein_pair",
        nargs=4,
        metavar=("TAR_PROTEIN", "TAR_CHAIN", "SRC_PROTEIN", "SRC_CHAIN"),
        help="Specify a single protein pair instead of a CSV."
    )
    group.add_argument(
        '--protein_database_search',
        nargs=3,
        metavar=("SRC_PROTEIN","SRC_CHAIN","DATABASE"),
        help="Search a specific protein pair against a database of protein chains (.csv file)"
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
        calibration_model_path= os.path.join( os.path.dirname(args.checkpoint)  , 'calibration_model.pkl' ),
        ligand_id=args.ligand_id,
        base_save_dir=args.base_save_dir,
        csv_path=args.csv_path,
        protein_pair=args.protein_pair,
        protein_database_search=args.protein_database_search,
        max_pLRMSD = args.max_pLRMSD,
        src_motif=args.src_motif,
        tar_motif=args.tar_motif
    )
    
    # Run the complete inference pipeline
    runner.run()


if __name__ == "__main__":
    main()
