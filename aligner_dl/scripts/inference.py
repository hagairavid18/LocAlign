import os
import subprocess
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml


sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'miners'))
sys.path.append('/home/iscb/wolfson/hagairavid/ScanNet_Ub')
from models.utils.collate import custom_collate_fn
from models.utils.misc import build_object
from miners.objects import Protein  # Adjust path if necessary
from datasets import ScanNetDataset  # Ensure this is in your PYTHONPATH


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using a trained model.")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/small-ligands-recycling1-baseline-ln-fast-10gnn-esm18/epoch=6-step=60984.ckpt",
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
        help="The ligand pdb name, if both inputs binds the same one"
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

    parser.add_argument(
        "--base_save_dir",
        type=str,
        default="inference_results",
        help="Directory to save inference results."
    )

    return parser.parse_args()


def save_non_ligand_models(df, output_dir) -> None:
    for _, row in df.iterrows():
        for protein, chain, ligand in [(row['ref_protein'], row['ref_chain'], row['ligand']), (row['mov_protein'],row['mov_chain'], row['ligand'])]: # TODO: handle multiple chains
            Protein(pdb_name=protein, chain_id=chain, ligand_name=ligand, save_models=True, ligand_dir=output_dir)

def run_scannet(df, output_dir: str, scannet_dir: str) -> None:
    """
    Runs ScanNet feature extraction using hardcoded Python from the 'py_scannet' conda environment.
    """
    script_path = "/home/iscb/wolfson/hagairavid/ScanNet_Ub/run_scannet.py"
    scannet_python = "/home/iscb/wolfson/hagairavid/miniforge3/envs/miner/envs/py_scannet_keras3/bin/python"  # Adjust to your system
    os.makedirs(scannet_dir, exist_ok=True)

    all_paths = []
    for idx, row in df.iterrows():
        if not os.path.exists(os.path.join(scannet_dir, row['ligand'], f"{row['ref_protein']}{row['ref_chain']}_scannet_atoms.pkl")):
            all_paths.append(os.path.join(output_dir, row['ligand'], f"{row['ref_protein']}{row['ref_chain']}_non_ligand_.ent"))
        if not os.path.exists(os.path.join(scannet_dir,row['ligand'], f"{row['mov_protein']}{row['mov_chain']}_scannet_atoms.pkl")):
            all_paths.append(os.path.join(output_dir, row['ligand'], f"{row['mov_protein']}{row['mov_chain']}_non_ligand_.ent"))

    cmd = [
        scannet_python,
        script_path,
        "--pdb_paths", *all_paths,
        "--output_dir", scannet_dir
    ]
    if len(all_paths) == 0:
        print("❗ No new structures to process. Skipping ScanNet feature extraction.")
        return
    print(f"🔄 Running ScanNet with: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, text=True, stdout=sys.stdout, stderr=sys.stderr)
        print(f"✅ ScanNet features saved in: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"❌ ScanNet feature extraction failed: {e}")


def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not args.csv_path:
    #     df = pd.read_csv(args.csv_path)
    #     csv_path = args.csv_path
    #     # rename columns to match expected format
    #     df.rename(columns={'Ligand_ID': 'ligand'}, inplace=True)
    # else:
        # Build DataFrame manually from provided pair
        ref, ref_chain, mov, mov_chain = args.protein_pair
        df = pd.DataFrame([{
        "ref_protein": ref,
        "mov_protein": mov,
        "ref_chain": ref_chain, 
        "mov_chain": mov_chain,
        "cath_degree": -1,
        "Ligand RMSD": 0.0,
        "ligand": args.ligand_id  # Or modify as needed
    }])
        csv_path = "temp_csv.csv"
    else:
        df = pd.read_csv(args.csv_path)

        # rename columns to match expected format
        df.rename(columns={'Ligand_ID': 'ligand'}, inplace=True)

        csv_path = args.csv_path
    df.to_csv(csv_path, index=False)


    experiment_name = args.checkpoint.split('/')[1]
    output_dir = os.path.join(args.base_save_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    # Step 1: Save non-ligand models
    checkpoint_dir = os.path.join("checkpoints", experiment_name)
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    print("Saving non-ligand models...")
    save_non_ligand_models(df, args.base_save_dir)

    # Step 2: Run ScanNet feature extraction
    print("Running ScanNet feature extraction...")
    run_scannet(df, args.base_save_dir, args.scannet_dir)

    # Step 3: Build dataset and dataloader
    print("Preparing dataset and dataloader...")
    dataset_config_path = os.path.join(checkpoint_dir, "dataset_config.yaml")
    if not os.path.isfile(dataset_config_path):
        raise FileNotFoundError(f"Model config file not found in checkpoint dir: {dataset_config_path}")
    with open(dataset_config_path) as f:
        dataset_config = yaml.safe_load(f)
    
    dataset_config['args']['df_path'] = csv_path
    dataset_config['args']['base_data_path'] = args.base_save_dir
    dataset_config['args']['base_embedding_path'] = args.scannet_dir
    dataset_config['args']['inference'] = True
    dataset_config['args']['ligand_column'] = 'ligand'
    
    dataset = ScanNetDataset(**dataset_config['args'])
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=custom_collate_fn, pin_memory=True)

    # Step 4: Build model and load checkpoint

    # Load model config YAML saved in this directory
    model_config_path = os.path.join(checkpoint_dir, "model_config.yaml")
    if not os.path.isfile(model_config_path):
        raise FileNotFoundError(f"Model config file not found in checkpoint dir: {model_config_path}")
   

    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)
    

    # Build model from saved config
    model = build_object(model_config, 'models')
    print("Loading model checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict)

    # Move model to device and eval
    model.to(device)
    model.eval()

    python_path = "/home/iscb/wolfson/hagairavid/miniforge3/envs/miner/envs/py_scannet_keras3/bin/python"  # Adjust to your system"
    script_path = "/home/iscb/wolfson/hagairavid/LocAlign/miners/scripts/chimera_pocket_viz.py"

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Running inference"):
            preds = model.inference_step(batch)

            metadata = preds["metadata"][0]

            # get all correspondences that above 0.5 of the highest correlation value
            threshold = 0.25 * torch.max(preds["corr_values"][0])
            above_threshold_indices = torch.where(preds["corr_values"][0] >= threshold)[0]

            # assert that there aer at least 3 correspondences
            if len(above_threshold_indices) < 3:
                above_threshold_indices = torch.topk(preds["corr_values"][0], 3)[1]
            print(f"Batch {idx}: Found {len(above_threshold_indices)} correspondences above threshold {threshold.item():.4f}")

            # top_k_corr_indices= torch.topk(preds["corr_values"][0], 10)[1]
            top_corr_values = preds["corr_values"][0][above_threshold_indices]  # Get top 10 correlation values
            top_corr_indices = preds["corr_indices"][0][above_threshold_indices]  # Get top 10 correlation values]
            top_corr_indices_atom = preds["corr_atom_indices"][0][above_threshold_indices]  # Get top 10 correlation values]
            ref = metadata["ref_protein"]
            mov = metadata["mov_protein"]
            ligand = metadata.get("ligand", "general")

            if not (ref and mov and ligand):
                print(f"⚠️ Skipping visualization for batch {idx} due to missing metadata")
                continue

            base_folder = os.path.join(output_dir, f"{ref}_{mov}_cath{metadata['cath_degree']}_{ligand}_rmsd{metadata.pop('Ligand RMSD', 0):.1f}_corr{preds['loss_dict']['corr_rmsd'].item():.2f}_gap{preds['loss_dict']['gap'].item():.2f}_emb{preds['loss_dict']['embedding'].item():.2f}")
            os.makedirs(base_folder, exist_ok=True)

            trans_dict = preds["transformation_dict"]

            # correspondences handling
            model_output_path = os.path.join(base_folder, f"{mov}_{ref}_output.npz")
            np.savez_compressed(model_output_path, 
                                top_corr_values=top_corr_values.numpy(), 
                                top_corr_indices=top_corr_indices.numpy(), 
                                top_corr_indices_atom=top_corr_indices_atom.numpy(),
                                R=trans_dict['pred_R'][0].numpy(), t=trans_dict['pred_t'][0].numpy())

            try:
                cmd = [
                    python_path,
                    script_path,
                    "--base_folder", base_folder,
                    "--scannet_dir", args.scannet_dir,
                    "--model_output_path", model_output_path,
                    "--template", ref + metadata['ref_chain'],
                    "--template_ligand", ligand,
                    "--query", mov + metadata['mov_chain']
                ]
                print(f"Running visualization: {' '.join(cmd)}")

                # Run without stopping on any error but print stdeout and stderr
                # result = subprocess.run(cmd, check=False, capture_output=True, text=True)
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    print(f"⚠️ Visualization script failed with code {result.returncode}")
                    print(f"stderr:\n{result.stderr}")
                else:
                    print("✅ Visualization completed successfully.")
            except Exception as e:
                print(f"Unexpected error during visualization: {e}")


    print("✅ All predictions processed and visualized.")
    print("✅ All predicted pairs processed and visualized.")


if __name__ == "__main__":
    main()
