"""
Materialize dataset artifacts (non-ligand PDB, ligand PDB, ScanNet features) for every
accepted apo structure found by miners/scripts/find_apo_structures.py, so ScanNetDataset
can read them by the same filename-token convention it uses for real PDB entries. See
the approved plan at /home/iscb/wolfson/hagairavid/.claude/plans/gleaming-greeting-stallman.md.

Usage (localign_infer3 env, CPU-only - forces CUDA_VISIBLE_DEVICES="" itself,
see below, since ScanNet feature extraction needs no GPU):
    python miners/scripts/materialize_apo_artifacts.py \
        --lookup_csv datasets/apo_structures/apo_lookup_results.csv \
        --scannet_dir /home/iscb/wolfson/hagairavid/scannet_2212
"""
import argparse
import logging
import os
import sys
from collections import namedtuple

# Must be set before torch (imported transitively via scripts.run_scannet's keras
# import below) initializes CUDA - a shell/srun-level `export CUDA_VISIBLE_DEVICES=""`
# was NOT enough to keep this process off the GPU (confirmed: it still grabbed a
# GPU and crashed with a CUDA OOM colliding with another job's memory, twice, even
# with the export in place in the .slurm script). Setting it here, before any
# CUDA-aware import, is the only place guaranteed to actually take effect.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd

sys.path.append(os.getcwd())

from miners.objects.protein import Protein
from miners.utils.constants import LIGAND_DIR
from scripts.run_scannet import extract_scannet

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_SelfPair = namedtuple("_SelfPair", ["tar_protein", "tar_chain", "src_protein", "src_chain", "ligand", "tar_ligand", "src_ligand"])


def materialize_non_ligand_and_ligand_files(lookup_df: pd.DataFrame, ligand_dir: str) -> list[_SelfPair]:
    """Build (non_ligand.ent, ligand.pdb) artifacts for every accepted apo result via
    Protein.__init__ (unchanged), and return the corresponding self-pairs for ScanNet."""
    pairs = []
    for _, row in lookup_df[lookup_df["status"] == "found"].iterrows():
        try:
            Protein(
                pdb_name=row["local_pdb_path"],
                chain_id=row["apo_chain_id"],
                ligand_name=row["ligand_id"],
                pdb_id=row["synthetic_id"],
                ligand_dir=ligand_dir,
                save_models=True,
            )
        except Exception as e:
            logger.error(f"Failed to materialize artifacts for {row['synthetic_id']}: {e}")
            continue

        pairs.append(_SelfPair(
            tar_protein=row["synthetic_id"], tar_chain=row["apo_chain_id"],
            src_protein=row["synthetic_id"], src_chain=row["apo_chain_id"],
            ligand=row["ligand_id"], tar_ligand=row["ligand_id"], src_ligand=row["ligand_id"],
        ))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize ScanNet/dataset artifacts for accepted apo structures")
    parser.add_argument("--lookup_csv", required=True, help="apo_lookup_results.csv from find_apo_structures.py")
    parser.add_argument("--ligand_dir", default=LIGAND_DIR, help="Base ligand dir (non_ligand/ligand pdb files)")
    parser.add_argument("--scannet_dir", required=True, help="base_scannet_path matching the target checkpoint's dataset_config.yaml")
    args = parser.parse_args()

    lookup_df = pd.read_csv(args.lookup_csv, dtype=str)
    n_found = (lookup_df["status"] == "found").sum()
    logger.info(f"{n_found}/{len(lookup_df)} lookups accepted; materializing artifacts for those")

    pairs = materialize_non_ligand_and_ligand_files(lookup_df, args.ligand_dir)
    extract_scannet(pairs, args.ligand_dir, args.scannet_dir)


if __name__ == "__main__":
    main()
