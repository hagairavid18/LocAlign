"""Report exact per-partition counts and train/val leakage checks for a pair-manifest split.

For each of {train,val,test}.csv under --split-dir, reports: number of pairs, number of unique
(protein, chain) identifiers, number of unique ligands, number of unique sequence clusters (if
the manifest has tar_cluster/src_cluster columns), and number of unique CATH superfamily groups
(via the raw CATH domain classification file). Also reports train-vs-val overlap on each of those
axes, to make explicit which overlaps a given split is designed to eliminate (e.g. the homology
split zeroes out chain/cluster overlap; the ligand split zeroes out ligand overlap) versus which
overlaps are expected and not a leakage concern.

CATH domain list format: https://download.cathdb.info/cath/releases/ ("CATH List File").
"""
import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_cath_lookup(cath_domain_list_path: Path) -> dict[str, set[tuple[str, str, str, str]]]:
    """chain_id (lowercase pdbcode + chain letter) -> set of (C,A,T,H) superfamily tuples."""
    lookup = defaultdict(set)
    with open(cath_domain_list_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            domain = parts[0]
            pdb_code, chain = domain[0:4].lower(), domain[4]
            lookup[f"{pdb_code}{chain}"].add(tuple(parts[1:5]))
    return lookup


def load_partition(path: Path, cath_lookup: dict) -> dict | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, dtype=str)
    except pd.errors.EmptyDataError:
        return None

    chains = set(zip(df["tar_protein"].str.lower(), df["tar_chain"])) | set(
        zip(df["src_protein"].str.lower(), df["src_chain"])
    )
    ligands = set(df["ligand_id"].dropna())
    clusters = None
    if "tar_cluster" in df.columns:
        clusters = set(df["tar_cluster"].dropna()) | set(df["src_cluster"].dropna())

    cath_groups, matched = set(), 0
    for protein, chain in chains:
        hits = cath_lookup.get(f"{protein}{chain}")
        if hits:
            matched += 1
            cath_groups |= hits

    return dict(
        n_pairs=len(df), chains=chains, ligands=ligands, clusters=clusters,
        cath_groups=cath_groups, n_chains_matched=matched,
    )


def report_split(split_dir: Path, cath_lookup: dict) -> None:
    partitions = {}
    for part in ["train", "val", "test"]:
        partitions[part] = load_partition(split_dir / f"{part}.csv", cath_lookup)

    print(f"=== {split_dir.name} ===")
    for part, s in partitions.items():
        if s is None:
            print(f"  {part:6s}: empty")
            continue
        clusters_str = "n/a" if s["clusters"] is None else len(s["clusters"])
        print(
            f"  {part:6s}: pairs={s['n_pairs']:>7} | unique chains={len(s['chains']):>6} "
            f"(CATH-matched {s['n_chains_matched']}) | unique ligands={len(s['ligands']):>4} | "
            f"seq clusters={clusters_str} | unique CATH (C.A.T.H) groups={len(s['cath_groups'])}"
        )

    train, val = partitions["train"], partitions["val"]
    if train and val:
        chain_overlap = len(train["chains"] & val["chains"])
        ligand_overlap = len(train["ligands"] & val["ligands"])
        print(f"  leakage check (train vs val):")
        print(f"    chain overlap:   {chain_overlap} / {len(val['chains'])} val chains also in train")
        print(f"    ligand overlap:  {ligand_overlap} / {len(val['ligands'])} val ligands also in train")
        if train["clusters"] is not None and val["clusters"] is not None:
            cluster_overlap = len(train["clusters"] & val["clusters"])
            print(f"    cluster overlap: {cluster_overlap} / {len(val['clusters'])} val clusters also in train")
        else:
            print("    cluster overlap: n/a (no cluster columns in this split)")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--splits-root", type=Path, default=Path("datasets/csv_files"),
        help="Directory containing one subdirectory per split (each with train/val/test.csv).",
    )
    ap.add_argument(
        "--split", action="append", dest="splits", default=None,
        help="Split subdirectory name under --splits-root (repeatable). Default: homology_25_10, ligand_25_10.",
    )
    ap.add_argument(
        "--cath-domain-list", type=Path, required=True,
        help="Path to the raw CATH domain list file (see docs/REPRODUCIBILITY.md#version-pins for the download link).",
    )
    args = ap.parse_args()
    splits = args.splits or ["homology_25_10", "ligand_25_10"]

    cath_lookup = load_cath_lookup(args.cath_domain_list)
    print(f"Loaded CATH domain list: {len(cath_lookup)} unique (pdb+chain) entries\n")

    for split in splits:
        report_split(args.splits_root / split, cath_lookup)
