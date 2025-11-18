#!/usr/bin/env python3
"""
Simple helper: read a CSV of pairs and run ScanNet feature extraction
using the existing `extract_scannet` helper in `scripts/run_scannet.py`.

Usage:
  python miners/run_scannet_from_csv.py --csv pairs.csv --pdb-dir /path/to/pdbs --scannet-dir /path/to/scannet_out
"""
import os
import sys
import argparse
from dataclasses import dataclass
from typing import Iterable

# Make repo root importable regardless of current working directory
script_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(os.getcwd())
from scripts.run_scannet import extract_scannet



@dataclass
class Pair:
    ref_protein: str
    ref_chain: str
    mov_protein: str
    mov_chain: str
    ligand: str


def read_pairs_from_csv(csv_path: str,
                        ligand_col: str = 'Ligand_ID',
                        ref_protein_col: str = 'ref_protein',
                        ref_chain_col: str = 'ref_chain',
                        mov_protein_col: str = 'mov_protein',
                        mov_chain_col: str = 'mov_chain') -> list[Pair]:
    """Read CSV and return list of Pair objects (uses pandas for robustness)."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    pairs: list[Pair] = []
    for _, row in df.iterrows():
        pairs.append(Pair(
            ref_protein=str(row[ref_protein_col]),
            ref_chain=str(row[ref_chain_col]),
            mov_protein=str(row[mov_protein_col]),
            mov_chain=str(row[mov_chain_col]),
            ligand=str(row[ligand_col])
        ))
    return pairs


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ScanNet on structures listed in a CSV file")
    parser.add_argument('--csv', required=True, help='CSV file with pairs')
    parser.add_argument('--pdb-dir', required=True, help='Base directory containing per-ligand PDB files')
    parser.add_argument('--scannet-dir', required=True, help='Output directory for ScanNet .pkl files')
    parser.add_argument('--atom-types', action='store_true', help='Compute and include atom type labels in output')
    parser.add_argument('--ligand-col', default='Ligand_ID')
    parser.add_argument('--ref-protein-col', default='ref_protein')
    parser.add_argument('--ref-chain-col', default='ref_chain')
    parser.add_argument('--mov-protein-col', default='mov_protein')
    parser.add_argument('--mov-chain-col', default='mov_chain')

    args = parser.parse_args(list(argv) if argv is not None else None)

    pairs = read_pairs_from_csv(
        args.csv,
        ligand_col=args.ligand_col,
        ref_protein_col=args.ref_protein_col,
        ref_chain_col=args.ref_chain_col,
        mov_protein_col=args.mov_protein_col,
        mov_chain_col=args.mov_chain_col,
    )

    # Ensure output dir exists
    os.makedirs(args.scannet_dir, exist_ok=True)

    # Delegate to extract_scannet which will skip existing caches
    extract_scannet(pairs, pdb_dir=args.pdb_dir, scannet_dir=args.scannet_dir)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
