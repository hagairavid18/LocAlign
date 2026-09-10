"""Materialize the ligand atom-name correspondence used for ligand_rmsd.

Reimplements the shared-atom-name intersection that is otherwise recomputed
on the fly in aligner_dl/datasets/scannet_dataset.py (PairDataset._read_ligand
and __getitem__, ~lines 125-132), so the exact atom mapping behind reported
evaluation numbers can be inspected/reproduced without rerunning training
code against the raw ligand PDB files.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from Bio.PDB import PDBParser

# Kept as a literal rather than imported from aligner_dl.utils.constants: this
# script's callers may have an unrelated third-party "utils" module earlier on
# sys.path, which shadows aligner_dl/utils as a namespace package.
DEFAULT_LIGAND_DIR = "/home/iscb/wolfson/hagairavid/ligands_25_10_2025"

_parser = PDBParser(QUIET=True)


def read_ligand_atom_ids(ligand_dir: Path, ligand_id: str, chain: str):
    ligand_path = ligand_dir / str(ligand_id) / f"{chain}_ligand.pdb"
    if not ligand_path.exists():
        return None
    structure = _parser.get_structure(chain, str(ligand_path))
    chain_obj = list(list(structure)[0])[0]
    return [atom.id for residue in chain_obj for atom in residue]


def build_mapping(manifest_csv: Path, ligand_dir: Path, out_path: Path) -> None:
    df = pd.read_csv(manifest_csv)
    n_ok = n_missing = n_empty = 0
    with open(out_path, "w") as f:
        for _, row in df.iterrows():
            ligand_id = row["ligand_id"]
            tar_chain = f"{row['tar_protein']}{row['tar_chain']}"
            src_chain = f"{row['src_protein']}{row['src_chain']}"
            tar_ids = read_ligand_atom_ids(ligand_dir, ligand_id, tar_chain)
            src_ids = read_ligand_atom_ids(ligand_dir, ligand_id, src_chain)
            if tar_ids is None or src_ids is None:
                n_missing += 1
                continue
            shared = sorted(set(src_ids) & set(tar_ids))
            if not shared:
                n_empty += 1
            record = {
                "ligand_id": ligand_id,
                "tar_protein": row["tar_protein"], "tar_chain": row["tar_chain"],
                "src_protein": row["src_protein"], "src_chain": row["src_chain"],
                "tar_n_atoms": len(tar_ids), "src_n_atoms": len(src_ids),
                "n_shared_atoms": len(shared),
                "shared_atom_names": shared,
            }
            f.write(json.dumps(record) + "\n")
            n_ok += 1
    print(f"{out_path}: {n_ok} pairs mapped, {n_missing} missing ligand PDBs, {n_empty} with zero shared atoms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path, help="Pair manifest CSV (e.g. datasets/csv_files/homology_25_10/val.csv)")
    ap.add_argument("--ligand-dir", default=Path(DEFAULT_LIGAND_DIR), type=Path, help="Directory of per-ligand-id ligand-only PDB files (aligner_dl.utils.constants.LIGAND_DIR)")
    ap.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_mapping(args.manifest, args.ligand_dir, args.output)
