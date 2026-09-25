"""
Report exact per-partition counts (pairs, unique chains, ligands, sequence clusters, CATH
groups) and train/val leakage checks for both released splits (Reviewer 1, comment 5).

Joins each split's manifests against the raw CATH domain classification file for per-chain
CATH (Class.Architecture.Topology.Homology) group labels, using the same chain-key convention
and domain->chain join as miners/parsers/biolip_reader.ipynb's compute_cath_degree() cell (a
chain key is f"{pdb_id}{chain_id}", matched against cath-domain-list.txt's pdb_chain field with
its trailing 2-digit domain code stripped).

Usage:
    python scripts/report_partition_stats.py \
        --cath_domain_list cath-classification-data/cath-domain-list.txt \
        --csv_dirs datasets/csv_files/homology_25_10 datasets/csv_files/ligand_25_10 \
        --out_dir results/partition_stats
"""
import argparse
import os

import pandas as pd

PARTITIONS = ["train", "val", "test"]


def load_cath_chain_groups(cath_domain_list_path: str) -> dict[str, set[str]]:
    """Return a dict mapping chain key (e.g. '1oaiA') -> set of CATH group ('C.A.T.H')
    labels of every domain CATH assigns to that chain. A chain commonly spans more than one
    domain (~36% of chains in v4.4.0 have 2+), each independently classified, so all of a
    chain's domain groups are kept rather than collapsing to one (matches
    compute_cath_degree()'s any-domain-of-the-chain comparison, generalized to counting)."""
    cath_chains = pd.read_csv(
        cath_domain_list_path, sep=r"\s+",
        names=["pdb_chain", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
        skiprows=16,
    )
    cath_chains["chain"] = cath_chains["pdb_chain"].str[:-2]
    cath_chains["cath_group"] = (
        cath_chains["a"].astype(str) + "."
        + cath_chains["b"].astype(str) + "."
        + cath_chains["c"].astype(str) + "."
        + cath_chains["d"].astype(str)
    )
    return cath_chains.groupby("chain")["cath_group"].apply(set).to_dict()


_EMPTY_COLUMNS = ["tar_protein", "tar_chain", "src_protein", "src_chain", "ligand_id"]


def load_partition(csv_dir: str, partition: str) -> pd.DataFrame:
    """Load one partition CSV, normalizing to an empty (0-row) frame with the expected
    columns when the file is missing, blank, or (as with ligand_25_10/test.csv) a stale
    header in an old column-naming convention with no data rows - all of these mean the
    same thing here: this partition is unused, as documented in docs/REPRODUCIBILITY.md."""
    path = os.path.join(csv_dir, f"{partition}.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=_EMPTY_COLUMNS)
    try:
        df = pd.read_csv(path, dtype=str)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=_EMPTY_COLUMNS)
    if len(df) == 0 or not set(_EMPTY_COLUMNS) <= set(df.columns):
        return pd.DataFrame(columns=_EMPTY_COLUMNS)
    return df


def chain_keys(df: pd.DataFrame) -> set[str]:
    """Unique (protein, chain) keys across both sides of every pair, as 'pdbidCHAIN'."""
    tar = set((df["tar_protein"] + df["tar_chain"]).tolist())
    src = set((df["src_protein"] + df["src_chain"]).tolist())
    return tar | src


def cluster_keys(df: pd.DataFrame) -> set[str] | None:
    if "tar_cluster" not in df.columns or "src_cluster" not in df.columns:
        return None
    return set(df["tar_cluster"].tolist()) | set(df["src_cluster"].tolist())


def cath_groups_for_chains(chains: set[str], chain_to_groups: dict[str, set[str]]) -> set[str]:
    groups = set()
    for chain in chains:
        groups |= chain_to_groups.get(chain, set())
    return groups


def summarize_split(csv_dir: str, chain_to_groups: dict[str, set[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    partitions = {p: load_partition(csv_dir, p) for p in PARTITIONS}
    partitions = {p: df for p, df in partitions.items() if df is not None}

    rows = []
    parsed = {}
    for name, df in partitions.items():
        chains = chain_keys(df) if len(df) else set()
        ligands = set(df["ligand_id"].tolist()) if len(df) else set()
        clusters = cluster_keys(df) if len(df) else None
        groups = cath_groups_for_chains(chains, chain_to_groups)
        parsed[name] = {"chains": chains, "ligands": ligands, "clusters": clusters}
        rows.append({
            "partition": name,
            "pairs": len(df),
            "unique_chains": len(chains),
            "unique_ligands": len(ligands),
            "sequence_clusters": len(clusters) if clusters is not None else "n/a",
            "cath_groups": len(groups),
        })
    counts_df = pd.DataFrame(rows)

    leak_rows = []
    if "train" in parsed and "val" in parsed:
        train, val = parsed["train"], parsed["val"]
        chain_overlap = len(val["chains"] & train["chains"])
        ligand_overlap = len(val["ligands"] & train["ligands"])
        if val["clusters"] is not None and train["clusters"] is not None:
            cluster_overlap = len(val["clusters"] & train["clusters"])
            cluster_str = f"{cluster_overlap}/{len(val['clusters'])}"
        else:
            cluster_str = "n/a"
        leak_rows.append({
            "chain_overlap": f"{chain_overlap}/{len(val['chains'])}",
            "ligand_overlap": f"{ligand_overlap}/{len(val['ligands'])}",
            "sequence_cluster_overlap": cluster_str,
        })
    leak_df = pd.DataFrame(leak_rows)
    return counts_df, leak_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cath_domain_list", default="cath-classification-data/cath-domain-list.txt")
    parser.add_argument("--csv_dirs", nargs="+",
                         default=["datasets/csv_files/homology_25_10", "datasets/csv_files/ligand_25_10"])
    parser.add_argument("--out_dir", default="results/partition_stats")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    chain_to_groups = load_cath_chain_groups(args.cath_domain_list)

    for csv_dir in args.csv_dirs:
        split_name = os.path.basename(os.path.normpath(csv_dir))
        counts_df, leak_df = summarize_split(csv_dir, chain_to_groups)

        print(f"\n{'='*60}\n{split_name}\n{'='*60}")
        print(counts_df.to_string(index=False))
        if not leak_df.empty:
            print("\nTrain/val leakage (overlap/val_total):")
            print(leak_df.to_string(index=False))

        counts_path = os.path.join(args.out_dir, f"{split_name}_partition_counts.csv")
        leak_path = os.path.join(args.out_dir, f"{split_name}_leakage_check.csv")
        counts_df.to_csv(counts_path, index=False)
        leak_df.to_csv(leak_path, index=False)
        print(f"\nWrote {counts_path}\nWrote {leak_path}")


if __name__ == "__main__":
    main()
