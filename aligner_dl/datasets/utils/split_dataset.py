import json
import pandas as pd
import argparse
import os
import numpy as np
import subprocess
import random
from collections import defaultdict
import warnings

from utils.constants import LIGAND_DIR
from utils.misc import deserialize_nested_lists

from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB import PDBParser, PPBuilder

warnings.simplefilter('ignore', PDBConstructionWarning)



def extract_sequence_from_pdb(pdb_file: str, chain_id: str =None) -> str:
    """
    Extracts the amino acid sequence from a PDB file for a given chain.
    """
    parser = PDBParser(QUIET=True)
    ppb = PPBuilder()

    structure = parser.get_structure("pdb", pdb_file)
    model = structure[0]

    # Select the specified chain, or the first one if not provided
    if chain_id:
        try:
            chain = model[chain_id]
        except KeyError:
            raise ValueError(f"Chain {chain_id} not found in {pdb_file}")
    else:
        chain = list(model)[0]

    peptides = ppb.build_peptides(chain)
    if not peptides:
        raise ValueError(f"No peptide chains found in {pdb_file}, chain {chain.id}")

    sequence = "".join(str(peptide.get_sequence()) for peptide in peptides)
    return sequence


def cluster_sequences(list_sequences, seqid=1.0, coverage=0.8, covmode='0', path2mmseqstmp='/tmp', path2mmseqs='/home/iscb/wolfson/hagairavid/miniforge3/envs/aligner_dl2/bin/mmseqs'):
    rng = np.random.randint(0, high=int(1e6))
    tmp_input = os.path.join(path2mmseqstmp, f'tmp_input_file_{rng}.fasta')
    tmp_output = os.path.join(path2mmseqstmp, f'tmp_output_file_{rng}')

    with open(tmp_input, 'w') as f:
        for k, sequence in enumerate(list_sequences):
            f.write(f'>{k}\n{sequence}\n')

    command = f'{path2mmseqs} easy-cluster {tmp_input} {tmp_output} {path2mmseqstmp} --min-seq-id {seqid} -c {coverage} --cov-mode {covmode}'
    subprocess.run(command.split(' '), check=True)
    print(f'command = {command}')

    with open(tmp_output + '_rep_seq.fasta', 'r') as f:
        representative_indices = [int(x[1:-1]) for x in f.readlines()[::2]]

    cluster_indices = np.zeros(len(list_sequences), dtype=int)
    table = pd.read_csv(tmp_output + '_cluster.tsv', sep='\t', header=None).to_numpy(dtype=int)
    for i, j in table:
        if i in representative_indices:
            cluster_indices[j] = representative_indices.index(i)

    for file in [tmp_output + '_rep_seq.fasta', tmp_output + '_all_seqs.fasta', tmp_output + '_cluster.tsv']:
        os.remove(file)

    return cluster_indices.tolist()

def hash_string(s: str) -> str:
    """
    Generate a hash for a given string.
    """
    import hashlib
    return hashlib.md5(s.encode()).hexdigest()

def split_csv(
    input_csv: str, 
    output_dir: str, 
    test_size: float = 0.2, 
    val_size: float = 0.1,
    mmseq_id_threshold: float = 0.6, 
    base_data_path: str = LIGAND_DIR,
    group_by_ligand: bool = False
    ) -> None:

    # Read the CSV file
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    # df = df[:50]
    print(f"Read CSV with {len(df)} rows")

    excluded_chunks = []

    dup_mask = df.duplicated(subset=['tar_protein', 'src_protein'], keep=False)
    if dup_mask.any():
        dup_excluded = df[dup_mask].copy()
        dup_excluded['exclusion_reason'] = 'duplicate_pair'
        excluded_chunks.append(dup_excluded)
    df = df[~dup_mask]
    print(f"Removed duplicates, {len(df)} rows remaining")
    # remove n_transforamtions != 1
    # df = df[df['n_transformations'] == 1]
    # print(f"Filtered by n_transformations == 1, {len(df)} rows remaining")
    failure_mask = df['failure_message'].notnull() & (df['failure_message'] != "")
    if failure_mask.any():
        failure_excluded = df[failure_mask].copy()
        failure_excluded['exclusion_reason'] = 'failure_message: ' + failure_excluded['failure_message'].astype(str)
        excluded_chunks.append(failure_excluded)
    df = df[~failure_mask]
    print(f"Filtered out failures, {len(df)} rows remaining")

    if excluded_chunks:
        excluded_log_path = os.path.join(output_dir, "excluded_pairs.csv")
        pd.concat(excluded_chunks, ignore_index=True).to_csv(excluded_log_path, index=False)
        print(f"Wrote {sum(len(c) for c in excluded_chunks)} excluded rows to {excluded_log_path}")

     # Ensure the "ligand_id" column exists if group_by_ligand is True
    if group_by_ligand and "ligand_id" not in df.columns:
        raise ValueError("The CSV file must contain a 'ligand_id' column when group_by_ligand is enabled")

    # Split based on ligand_id grouping if specified
    # for col in  ['bbr', 'bbc']:
    #     if df[col].apply(lambda x: isinstance(x, str) and x.startswith('[') and x.endswith(']')).any():
    #         df[col] = df[col].fillna('[]')
            
    #         df[col] = df[col].apply(lambda x: json.loads(x))
    #         df[col] = df[col].apply(lambda x: deserialize_nested_lists(x, col))
    
    # df['bbr'] = df['bbr'].apply(lambda x: max([max(y) for y in x if len(y) > 0], default=float('-inf')))
    # df['bbc'] = df['bbc'].apply(lambda x: max([max(y) for y in x if len(y) > 0], default=float('-inf')))
    # df = df[df['bbr'] > 0.3]
    # df = df[df['bbc'] > 10]
    # print(f"Filtered Best BBR and BBC, {len(df)} rows remaining")
    high_cath_degree = df[df['cath_degree'] > 3]
    print(f"Grouped by tar_protein and src_protein, {len(df)} rows remaining")
    high_cath_degree = high_cath_degree.groupby('ligand_id').head(1000)
    df = pd.concat([df[df['cath_degree'] < 4], high_cath_degree])
    print(f"Grouped by ligand_id, {len(df)} rows remaining")
    if group_by_ligand:
        # Get unique ligand_ids and shuffle
        ligand_ids = df['ligand_id'].unique()
        random.seed(42)  # For reproducibility
        random.shuffle(ligand_ids)

        # Calculate number of ligands for each split
        total_ligands = len(ligand_ids)
        test_len = int(total_ligands * test_size)
        val_len = int(total_ligands * val_size)

        # Assign ligand_ids to each split
        test_ligands = ligand_ids[:test_len]
        val_ligands = ligand_ids[test_len:test_len + val_len]
        train_ligands = ligand_ids[test_len + val_len:]

        # Create datasets for each split based on ligand_id
        train_df = df[df['ligand_id'].isin(train_ligands)]
        val_df = df[df['ligand_id'].isin(val_ligands)]
        val_df = val_df.groupby('ligand_id').head(300)  # Ensure each ligand is present in the validation set
        test_df = df[df['ligand_id'].isin(test_ligands)]
    else:

    
        sequence_list = []
        unique_triples = {}  # (ligand_id, protein_id, chain_id) -> sequence
        
        csv_hash = hash_string(input_csv)
        sequences_path = os.path.join("datasets" , "csv_files", f'sequences_{csv_hash}.txt')

        # Load cached sequences if available
        # if os.path.exists(sequences_path):
        #     print("Cached sequences found, loading from file")
        #     with open(sequences_path, 'r') as f:
        #         for line in f:
        #             parts = line.strip().split('\t')
        #             if len(parts) == 4:
        #                 ligand_id, protein_id, chain_id, sequence = parts
        #                 key = (ligand_id, protein_id, chain_id)
        #                 sequence = sequence.replace(' ', '').replace('\n', '')
        #                 unique_triples[key] = sequence
        #     sequence_list = list(unique_triples.values())
        #     print(f"Loaded {len(sequence_list)} cached sequences")

        # # Otherwise, extract sequences and cache them
        # else:
        print("No cached sequences found, extracting from PDB files")
        for row_idx, row in df.iterrows():
            for prot_col, chain_col in [('tar_protein', 'tar_chain'), ('src_protein', 'src_chain')]:
                protein_id = row[prot_col]
                chain_id = row[chain_col]
                ligand_id = row['ligand_id']
                key = (ligand_id, protein_id, chain_id)

                if key not in unique_triples:
                    pdb_file = os.path.join(base_data_path, ligand_id, f"{protein_id}{chain_id}_non_ligand_.ent")
                    try:
                        seq = extract_sequence_from_pdb(pdb_file, chain_id=chain_id)
                        if seq:
                            seq = seq.replace(' ', '').replace('\n', '')
                            unique_triples[key] = seq
                    except Exception as e:
                        print(f"[Warning] Skipping {protein_id} chain {chain_id} in ligand {ligand_id}: {e}")

        # Write sequences to cache
        with open(sequences_path, 'w') as f:
            for (ligand_id, protein_id, chain_id), seq in unique_triples.items():
                f.write(f"{ligand_id}\t{protein_id}\t{chain_id}\t{seq}\n")
        sequence_list = list(unique_triples.values())
        print(f"Extracted and cached {len(sequence_list)} sequences")

        # Cluster sequences
        cluster_ids = cluster_sequences(sequence_list, seqid=mmseq_id_threshold, coverage=0.6)
        triple_to_cluster = {triple: cluster_ids[i] for i, triple in enumerate(unique_triples.keys())}

        # Assign clusters to df
        df['tar_cluster'] = np.nan
        df['src_cluster'] = np.nan

        for row_idx, row in df.iterrows():
            for cluster_col, prot_col, chain_col in [
                ('tar_cluster', 'tar_protein', 'tar_chain'),
                ('src_cluster', 'src_protein', 'src_chain')
            ]:
                ligand_id = row['ligand_id']
                protein_id = row[prot_col]
                chain_id = row[chain_col]
                key = (ligand_id, protein_id, chain_id)

                cluster = triple_to_cluster.get(key)
                if cluster is not None:
                    df.at[row_idx, cluster_col] = cluster

        # Filter out rows where either protein has no cluster
        df = df.dropna(subset=['tar_cluster', 'src_cluster'])
        df['tar_cluster'] = df['tar_cluster'].astype(int)
        df['src_cluster'] = df['src_cluster'].astype(int)

        # Group by clusters
        cluster_to_proteins_and_chains = defaultdict(set)
        for (ligand_id, protein_id, chain_id), cluster_id in triple_to_cluster.items():
            cluster_to_proteins_and_chains[cluster_id].add((protein_id, chain_id))

        # Shuffle clusters for random splitting
        cluster_ids_unique = list(cluster_to_proteins_and_chains.keys())
        total_examples = 0
        while total_examples< 39000:
            random.shuffle(cluster_ids_unique)

            # Split clusters
            total = len(cluster_ids_unique)
            n_test = int(test_size * total)
            n_val = int(val_size * total)

            test_clusters = set(cluster_ids_unique[:n_test])
            val_clusters = set(cluster_ids_unique[n_test:n_test + n_val])
            train_clusters = set(cluster_ids_unique[n_test + n_val:])

            # Assign proteins to splits
            protein_to_split = {}
            for cluster in train_clusters:
                for protein, chain in cluster_to_proteins_and_chains[cluster]:
                    protein_to_split[protein + chain] = 'train'
            for cluster in val_clusters:
                for protein, chain in cluster_to_proteins_and_chains[cluster]:
                    protein_to_split[protein + chain] = 'val'
            for cluster in test_clusters:
                for protein,chain in cluster_to_proteins_and_chains[cluster]:
                    protein_to_split[protein + chain] = 'test'

            # Only keep rows where both proteins are from same split
            split_dfs = {'train': [], 'val': [], 'test': []}
            for _, row in df.iterrows():
                tar_split = protein_to_split.get(row['tar_protein'] + row['tar_chain'])
                src_split = protein_to_split.get(row['src_protein'] + row['src_chain'])
                if tar_split is not None and tar_split == src_split:
                    split_dfs[tar_split].append(row)

            train_df = pd.DataFrame(split_dfs['train'])
            val_df = pd.DataFrame(split_dfs['val'])
            test_df = pd.DataFrame(split_dfs['test'])

            print(f"Train clusters: {len(train_clusters)}, rows: {len(train_df)}")
            print(f"Val clusters: {len(val_clusters)}, rows: {len(val_df)}")
            print(f"Test clusters: {len(test_clusters)}, rows: {len(test_df)}")
            total_examples = len(train_df) + len(val_df) + len(test_df)
            print(f"Total examples after split: {total_examples}")

    # Save the splits into CSV files
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"Data split completed. Files and histograms saved in {output_dir}")

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Split CSV by ligand_id column or regular row-based split")
    parser.add_argument('--input_csv', type=str, required=True, help="Path to the input CSV file")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output CSV files")
    parser.add_argument('--test_size', type=float, default=0.2, help="Fraction of the data to be used as the test set")
    parser.add_argument('--val_size', type=float, default=0.1, help="Fraction of the total data to be used as validation set")
    parser.add_argument('--mmseq_id_threshold', type=float, default=0.5, help="MMseqs2 sequence identity threshold for clustering")
    parser.add_argument('--group_by_ligand', action='store_true', help="Whether to group by ligand_id when splitting")


    args = parser.parse_args()

    # Call the function to split the CSV
    split_csv(args.input_csv, args.output_dir, test_size=args.test_size, val_size=args.val_size, base_data_path=LIGAND_DIR, mmseq_id_threshold=args.mmseq_id_threshold, group_by_ligand=args.group_by_ligand)
