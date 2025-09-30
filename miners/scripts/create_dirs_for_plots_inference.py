import json
import os
import pickle
import shutil
from glob import glob
import numpy as np
import sys

sys.path.append("/home/iscb/wolfson/hagairavid/LocAlign/aligner_dl")

from utils.constants import LIGAND_DIR
sys.path.append("/home/iscb/wolfson/hagairavid/LocAlign")
from miners.scripts.chimera_pocket_viz import process_alignment

# === CONFIG ===
PICKLE_DIR = "results/inference_outputs_top_validation"  # Path where each .pkl contains one prediction
OUTPUT_BASE_PATH = "inference_outputs_top_validation"  # Directory to organize output

json_files = sorted(glob(os.path.join(PICKLE_DIR, "*.json")))

for json_path in json_files:
    with open(json_path, "r") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    trans_dict = data.get("transformation_dict", {})

    template = metadata.get("ref_protein")
    query = metadata.get("mov_protein")
    # template = metadata.get("ref_protein") + '_' + metadata.get("ref_chain")
    # query = metadata.get("mov_protein") + '_' + metadata.get("mov_chain")
    ligand_id = metadata["ligand_id"]

    if not template or not query:
        print(f"⚠️ Skipping {pkl_path} — missing template/query names")
        continue

    # Create output folder
    pair_id = os.path.basename(json_path).replace(".json", "")
    pair_dir = os.path.join(OUTPUT_BASE_PATH, pair_id)
    os.makedirs(pair_dir, exist_ok=True)

    ligand_dir = os.path.join(LIGAND_DIR, 'general')

    # Copy only .cif files (no pockets, RANSAC, or ligand files)
    for protein in [template, query]:
        cif_file = os.path.join(ligand_dir, f"{protein}.cif")
        if os.path.exists(cif_file):
            shutil.copy(cif_file, pair_dir)
        else:
            print(f"⚠️ Missing .cif file: {cif_file}")

    # Combine R and t into a 4x4 transformation matrix
    try:
        R = np.array(trans_dict["pred_R"])  # shape: (3, 3)
        t = np.array(trans_dict["pred_t"]).reshape(3, 1)  # shape: (3, 1)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.squeeze()
    except Exception as e:
        print(f"❌ Failed to construct transformation matrix for {pair_id}: {e}")
        continue

    # Visualize using process_alignment
    process_alignment(
        base_folder=pair_dir,
        template=template + "_A",
        template_ligand=ligand_id,
        query=query + "_A",
        query_transformation=(R[0], t[:,0]),
    )


print("✅ All predicted pairs processed.")
