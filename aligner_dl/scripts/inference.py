import json
import os
import subprocess
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.constants import LIGAND_DIR
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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv_path", type=str, help="Path to CSV file with ref/mov chains.")
    group.add_argument("--protein_pair", nargs=2, metavar=("REF_PROTEIN", "MOV_PROTEIN"),
                       help="Specify a single protein pair instead of a CSV.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None, help="Device to run inference on.")
    parser.add_argument("--scannet_env", type=str, default="py_scannet", help="Conda environment name for ScanNet.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for DataLoader.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    return parser.parse_args()

def save_non_ligand_models(df, output_dir) -> None:
    for _, row in df.iterrows():
        for protein, chain in [(row['ref_protein'], 'A'), (row['mov_protein'], 'A')]:
            Protein(pdb_name=protein, chain_id=chain, ligand_name='general', save_models=True, ligand_dir=output_dir)

def run_scannet(df, output_dir: str, scannet_dir: str) -> None:
    """
    Runs ScanNet feature extraction using hardcoded Python from the 'py_scannet' conda environment.
    """
    script_path = "/home/iscb/wolfson/hagairavid/ScanNet_Ub/run_scannet.py"
    scannet_python = "/home/iscb/wolfson/hagairavid/miniforge3/envs/py_scannet/bin/python"  # Adjust to your system
    os.makedirs(scannet_dir, exist_ok=True)

    all_paths = []
    for idx, row in df.iterrows():
        if not os.path.exists(os.path.join(scannet_dir, 'general', f"{row['ref_protein']}_scannet_atoms.pkl")):
            all_paths.append(os.path.join(output_dir, 'general', f"{row['ref_protein']}_non_ligand.ent"))
        if not os.path.exists(os.path.join(scannet_dir,'general', f"{row['mov_protein']}_scannet_atoms.pkl")):
            all_paths.append(os.path.join(output_dir, 'general', f"{row['mov_protein']}_non_ligand.ent"))

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

def tensor_to_numpy(d):
    """Recursively convert torch.Tensors in dicts/lists/tuples to numpy arrays"""
    if isinstance(d, torch.Tensor):
        return d.cpu().numpy()
    elif isinstance(d, dict):
        return {k: tensor_to_numpy(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [tensor_to_numpy(v) for v in d]
    elif isinstance(d, tuple):
        return tuple(tensor_to_numpy(v) for v in d)
    return d
def main():
    args = parse_args()

    device = args.device
    if device is None:
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
        "ligand": "general"  # Or modify as needed
    }])
        csv_path = "temp_csv.csv"
        df.to_csv(csv_path, index=False)

    # Step 1: Save non-ligand models
    print("Saving non-ligand models...")
    save_non_ligand_models(df, args.output_dir)

    # Step 2: Run ScanNet feature extraction
    print("Running ScanNet feature extraction...")
    run_scannet(df, args.output_dir, args.scannet_dir)

    # Step 3: Build dataset and dataloader
    print("Preparing dataset and dataloader...")
    dataset = ScannetDataset(df_path=csv_path, base_data_path=args.output_dir, base_embedding_path=args.scannet_dir, level='atom', inference=True, ligand_column='ligand')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=custom_collate_fn, pin_memory=True)

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

    def tensor_to_list(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: tensor_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [tensor_to_list(x) for x in obj]
        else:
            return obj

    # Inside your saving loop:

    scannet_python = "/home/iscb/wolfson/hagairavid/miniforge3/envs/py_scannet/bin/python"
    script_path = "miners/scripts/chimera_pocket_viz.py"

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Running inference"):
            preds = model.inference_step(batch)

            metadata = preds.get("metadata", [{}])[0]
            ref = metadata.get("ref_protein")
            mov = metadata.get("mov_protein")
            ligand = metadata.get("ligand", "general")

            if not (ref and mov and ligand):
                print(f"⚠️ Skipping visualization for batch {idx} due to missing metadata")
                continue

            base_folder = os.path.join(args.output_dir, f"{ref}_{mov}")
            os.makedirs(base_folder, exist_ok=True)

            template = ref + "_A"
            query = mov + "_A"

            trans_dict = preds.get("transformation_dict", {})
            R = tensor_to_list(trans_dict.get("pred_R")[0])  # shape (3, 3)
            t = tensor_to_list(trans_dict.get("pred_t")[0])  # shape (3,)
            print(f"R: {R}, t: {t}")

            # Flatten R and round all values
            flat_R = np.round(sum(R, []), 6)  # shape (9,)
            t_vec = np.round(t, 6)            # shape (3,)
            transform_args = [f"{val:.6f}" for val in np.concatenate([flat_R, t_vec])]


            # Set paths
            scannet_python = "/home/iscb/wolfson/hagairavid/miniforge3/envs/py_scannet/bin/python"

            # Build the command
            cmd = [
                scannet_python,
                script_path,
                "--base_folder", base_folder,
                "--template", template ,
                "--template_ligand", ligand,
                "--query", query ,
                "--query_transformation"
            ] + transform_args


            print(f"Running visualization: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, text=True, stdout=sys.stdout, stderr=sys.stderr)

    print("✅ All predictions processed and visualized.")
    print("✅ All predicted pairs processed and visualized.")


if __name__ == "__main__":
    main()
