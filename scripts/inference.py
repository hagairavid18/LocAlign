import os
import sys
import argparse
from datetime import datetime
from typing import Any

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'aligner_dl'))

from aligner_dl.datasets import ScanNetDataset
from aligner_dl.models.utils.collate import custom_collate_fn
from aligner_dl.models.utils.misc import build_object
from miners.objects import Protein
from miners.scripts.chimera_pocket_viz import process_alignment
from scripts.run_scannet import extract_scannet


class InferenceRunner:
    """Handles inference pipeline for protein structure alignment."""
    
    def __init__(
        self,
        checkpoint_path: str,
        scannet_dir: str,
        ligand_id: str,
        base_save_dir: str,
        save_dir: str | None = None,
        csv_path: str | None = None,
        protein_pair: tuple[str, str, str, str] | None = None
    ) -> None:
        """
        Initialize the inference runner.
        
        Args:
            checkpoint_path (str): Path to model checkpoint
            scannet_dir (str): Directory for ScanNet pretrained embeddings
            ligand_id (str): Ligand PDB name
            base_save_dir (str): Base directory to save inference results
            save_dir (str | None): Specific directory name for this run (optional)
            csv_path (str | None): Path to CSV file with protein pairs (optional)
            protein_pair (tuple[str, str, str, str] | None): Single protein pair as 
                (ref, ref_chain, mov, mov_chain) (optional)
        """
        self._checkpoint_path = checkpoint_path
        self._scannet_dir = scannet_dir
        self._ligand_id = ligand_id
        self._base_save_dir = base_save_dir
        self._save_dir = save_dir
        self._csv_path = csv_path
        self._protein_pair = protein_pair
        
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._df = None
        self._csv_output_path = None
        self._experiment_name = None
        self._checkpoint_dir = None
        self._output_dir = None
        self._model = None
        self._dataloader = None
    
    def _prepare_dataframe(self) -> None:
        """
        Prepare dataframe from either CSV or protein pair.
        
        Returns:
            None: Sets self._df and self._csv_output_path
        """
        if not self._csv_path:
            ref, ref_chain, mov, mov_chain = self._protein_pair
            self._df = pd.DataFrame([{
                "ref_protein": ref,
                "mov_protein": mov,
                "ref_chain": ref_chain,
                "mov_chain": mov_chain,
                "cath_degree": -1,
                "Ligand RMSD": 0.0,
                "ligand": self._ligand_id
            }])
            self._csv_output_path = "temp_csv.csv"
        else:
            self._df = pd.read_csv(self._csv_path)
            # Rename columns to match expected format
            self._df.rename(columns={'Ligand_ID': 'ligand'}, inplace=True)
            self._csv_output_path = self._csv_path
        
        self._df.to_csv(self._csv_output_path, index=False)
    
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
        
        # Determine output directory
        if self._save_dir:
            self._output_dir = os.path.join(self._base_save_dir, self._save_dir)
        else:
            # Use experiment name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._output_dir = os.path.join(self._base_save_dir, f"{self._experiment_name}_{timestamp}")
        
        os.makedirs(self._output_dir, exist_ok=True)
        print(f"Saving results to: {self._output_dir}")
    
    def _save_non_ligand_models(self) -> None:
        """
        Save non-ligand PDB models for all proteins in the dataframe.
        Only saves unique protein-chain-ligand combinations to avoid duplicates.
        
        Returns:
            None: Downloads and saves PDB files to self._base_save_dir
        """
        # Collect unique protein-chain-ligand combinations using a set
        unique_combinations = set()
        for _, row in self._df.iterrows():
            unique_combinations.add((row['ref_protein'], row['ref_chain'], row['ligand']))
            unique_combinations.add((row['mov_protein'], row['mov_chain'], row['ligand']))
        
        # Save models with progress bar
        print(f"Saving {len(unique_combinations)} unique non-ligand models...")
        for protein, chain, ligand in tqdm(unique_combinations, desc="Downloading PDB files"):
            Protein(
                pdb_name=protein,
                chain_id=chain,
                ligand_name=ligand,
                save_models=True,
                ligand_dir=self._base_save_dir
            )
    
    def _extract_features(self) -> None:
        """
        Run ScanNet feature extraction.
        
        Returns:
            None: Generates feature files in self._scannet_dir
        """
        print("Running ScanNet feature extraction...")
        extract_scannet(self._df, self._base_save_dir, self._scannet_dir)
    
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
        
        # Update dataset config for inference
        dataset_config['args']['df_path'] = self._csv_output_path
        dataset_config['args']['base_data_path'] = self._base_save_dir
        dataset_config['args']['base_scannet_path'] = self._scannet_dir
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
            None: Sets self._model and moves it to device
        
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
            for idx, batch in tqdm(enumerate(self._dataloader), total=len(self._dataloader), desc="Running inference"):
                self._process_batch(idx, batch)
        
        print("✅ All predictions processed and visualized.")
    
    def _process_batch(self, idx: int, batch: Any) -> None:
        """
        Process a single batch and save results.
        
        Args:
            idx (int): Batch index
            batch (Any): Batch dictionary containing protein pair data
            
        Returns:
            None: Saves predictions and visualizations to self._output_dir
        """
        # Move tensors to device (model also handles this internally, but we keep it explicit here)
        batch = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Run model inference for this batch (batch_size is 1)
        preds = self._model.inference_step(batch)
        metadata = preds["metadata"][0]

        # Filter correspondences above threshold (25% of max correlation)
        corr_vals = preds["corr_values"][0]
        threshold = 0.25 * torch.max(corr_vals)
        above_threshold_indices = torch.where(corr_vals >= threshold)[0]

        # Ensure at least 3 correspondences
        if len(above_threshold_indices) < 3:
            above_threshold_indices = torch.topk(corr_vals, 3)[1]
        
        print(
            f"Batch {idx}: Found {len(above_threshold_indices)} correspondences "
            f"above threshold {threshold.item():.4f}"
        )

        top_corr_values = preds["corr_values"][0][above_threshold_indices]
        top_corr_indices = preds["corr_indices"][0][above_threshold_indices]
        top_corr_indices_atom = preds["corr_atom_indices"][0][above_threshold_indices]

        # Prepare save paths and folder name
        ref = metadata["ref_protein"]
        mov = metadata["mov_protein"]
        ligand = metadata.get("ligand", "general")

        ligand_rmsd = metadata.get('Ligand RMSD', 0)
        corr_rmsd = preds['loss_dict']['corr_rmsd'].item()
        gap = preds['loss_dict']['gap'].item()
        emb = preds['loss_dict']['embedding'].item()

        save_folder = os.path.join(
            self._output_dir,
            f"{ref}_{mov}_cath{metadata['cath_degree']}_{ligand}_"
            f"rmsd{ligand_rmsd:.1f}_corr{corr_rmsd:.2f}_gap{gap:.2f}_emb{emb:.2f}"
        )
        os.makedirs(save_folder, exist_ok=True)

        trans_dict = preds["transformation_dict"]

        # Convert to numpy arrays without saving to file
        R_np = trans_dict['pred_R'][0].detach().cpu().numpy()
        t_np = trans_dict['pred_t'][0].detach().cpu().numpy()
        
        # Generate visualization using Chimera via process_alignment (pass arrays directly)
        try:
            process_alignment(
                base_folder=save_folder,
                ligand=ligand,
                scannet_dir=self._scannet_dir,
                template=ref + metadata['ref_chain'],
                query=mov + metadata['mov_chain'],
                query_transformation=(R_np, t_np),
                corr_values=top_corr_values.detach().cpu().numpy(),
                corr_indices=top_corr_indices.detach().cpu().numpy(),
                atom_indexes_list=top_corr_indices_atom.detach().cpu().numpy()
            )
        except Exception as e:
            print(f"Unexpected error during visualization: {e}")
    
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
        self._prepare_dataloader()
        self._load_model()
        self._run_inference()


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using a trained model.")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/small-ligands-recycling2-corr01-embed2-gap01-lig5/epoch=9-step=87120.ckpt",
        help="Path to model checkpoint (default: preconfigured baseline)."
    )
    parser.add_argument(
        "--scannet_dir",
        type=str,
        default=os.getcwd(),
        help="Directory for saving ScanNet pretrained embedding."
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
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Specific directory name for this inference run. If not provided, will use experiment name with timestamp."
    )

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
    
    return parser.parse_args()


def main():
    """Main entry point for the inference script."""
    args = parse_args()
    
    # Initialize inference runner
    runner = InferenceRunner(
        checkpoint_path=args.checkpoint,
        scannet_dir=args.scannet_dir,
        ligand_id=args.ligand_id,
        base_save_dir=args.base_save_dir,
        save_dir=args.save_dir,
        csv_path=args.csv_path,
        protein_pair=args.protein_pair
    )
    
    # Run the complete inference pipeline
    runner.run()


if __name__ == "__main__":
    main()
