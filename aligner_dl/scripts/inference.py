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
from datasets import ScannetDataset  # Ensure this is in your PYTHONPATH

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using a trained model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment, i.e. checkpoint directory name under ./checkpoints/")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save inference results.")
    parser.add_argument("--scannet_dir", type=str, help="Directory containing ligand files.")
    parser.add_argument("--ligand_id", type=str, default="general", help="Ligand ID to use for non-ligand models.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv_path", type=str, help="Path to CSV file with ref/mov chains.")
    group.add_argument("--protein_pair", nargs=2, metavar=("REF_PROTEIN", "MOV_PROTEIN"),
                       help="Specify a single protein pair instead of a CSV.")
    return parser.parse_args()

def save_non_ligand_models(df, output_dir, ligand_id) -> None:
    for _, row in df.iterrows():
        for protein, chain in [(row['ref_protein'], 'A'), (row['mov_protein'], 'A')]:
            Protein(pdb_name=protein, chain_id=chain, ligand_name=ligand_id, save_models=True, ligand_dir=output_dir)

def run_scannet(df, output_dir: str, scannet_dir: str, ligand_id: str) -> None:
    """
    Runs ScanNet feature extraction using hardcoded Python from the 'py_scannet' conda environment.
    """
    script_path = "/home/iscb/wolfson/hagairavid/ScanNet_Ub/run_scannet.py"
    scannet_python = "/home/iscb/wolfson/hagairavid/miniforge3/envs/py_scannet/bin/python"  # Adjust to your system
    os.makedirs(scannet_dir, exist_ok=True)

    all_paths = []
    for idx, row in df.iterrows():
        if not os.path.exists(os.path.join(scannet_dir, ligand_id, f"{row['ref_protein']}_scannet_atoms.pkl")):
            all_paths.append(os.path.join(output_dir, ligand_id, f"{row['ref_protein']}_non_ligand.ent"))
        if not os.path.exists(os.path.join(scannet_dir,ligand_id, f"{row['mov_protein']}_scannet_atoms.pkl")):
            all_paths.append(os.path.join(output_dir, ligand_id, f"{row['mov_protein']}_non_ligand.ent"))

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

    if args.csv_path:
        df = pd.read_csv(args.csv_path)
        csv_path = args.csv_path
    else:
        # Build DataFrame manually from provided pair
        ref, mov = args.protein_pair
        df = pd.DataFrame([{
        "ref_protein": ref,
        "mov_protein": mov,
        "ligand": args.ligand_id  # Or modify as needed
    }])
        csv_path = "temp_csv.csv"
        df.to_csv(csv_path, index=False)

    # Step 1: Save non-ligand models
    print("Saving non-ligand models...")
    save_non_ligand_models(df, args.output_dir, args.ligand_id)

    # Step 2: Run ScanNet feature extraction
    print("Running ScanNet feature extraction...")
    run_scannet(df, args.output_dir, args.scannet_dir, args.ligand_id)

    # Step 3: Build dataset and dataloader
    print("Preparing dataset and dataloader...")
    dataset = ScannetDataset(df_path=csv_path, base_data_path=args.output_dir, base_embedding_path=args.scannet_dir, level='atom', inference=True, ligand_column='ligand')
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=custom_collate_fn, pin_memory=True)

    # Step 4: Build model and load checkpoint
    checkpoint_dir = os.path.join("checkpoints", args.experiment_name)
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

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

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    scannet_python = "/home/iscb/wolfson/hagairavid/miniforge3/envs/py_scannet/bin/python"
    script_path = "miners/scripts/chimera_pocket_viz.py"

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Running inference"):
            preds = model.inference_step(batch)

            metadata = preds["metadata"][0]
            top_k_corr_indices= torch.topk(preds["corr_values"][0], 10)[1]
            top_corr_values = preds["corr_values"][0][top_k_corr_indices]  # Get top 10 correlation values
            top_corr_indices = preds["corr_indices"][0][top_k_corr_indices]  # Get top 10 correlation values]
            ref = metadata["ref_protein"]
            mov = metadata["mov_protein"]
            ligand = metadata.get("ligand", "general")

            if not (ref and mov and ligand):
                print(f"⚠️ Skipping visualization for batch {idx} due to missing metadata")
                continue

            base_folder = os.path.join(args.output_dir, f"{ref}_{mov}")
            os.makedirs(base_folder, exist_ok=True)

            trans_dict = preds["transformation_dict"]

            # correspondences handling
            model_output_path = os.path.join(base_folder, f"{mov}_{ref}_output.npz")
            np.savez_compressed(model_output_path, top_corr_values=top_corr_values.numpy(), top_corr_indices=top_corr_indices.numpy(), R=trans_dict['pred_R'][0].numpy(), t=trans_dict['pred_t'][0].numpy())

            # Build the command
            cmd = [
                scannet_python,
                script_path,
                "--base_folder", base_folder,
                "--model_output_path", model_output_path,
                "--template", ref + "_A" ,
                "--template_ligand", ligand,
                "--query", mov + "_A" ]
          
            print(f"Running visualization: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, text=True, stdout=sys.stdout, stderr=sys.stderr)

    print("✅ All predictions processed and visualized.")
    print("✅ All predicted pairs processed and visualized.")


if __name__ == "__main__":
    main()
