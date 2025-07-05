import json
import pandas as pd
import argparse
import os
import numpy as np
import subprocess
import random
from utils.constants import LIGAND_DIR
from collections import defaultdict

from utils.misc import deserialize_nested_lists

from Bio.PDB import MMCIFParser, PPBuilder
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings

warnings.simplefilter('ignore', PDBConstructionWarning)


def extract_sequence_from_pdb(pdb_file: str) -> str: 
    """
    Extracts the amino acid sequence from a PDB file.
    """
    parser = MMCIFParser(QUIET=True)
    ppb = PPBuilder()

    structure = parser.get_structure("pdb", pdb_file)
    model = list(structure)[0]
    chain = list(model)[0]  # You can modify this to select by chain ID

    peptides = ppb.build_peptides(chain)
    if not peptides:
        raise ValueError(f"No peptide chains found in {pdb_file}")

    sequence = ""
    for peptide in peptides:
        sequence += str(peptide.get_sequence())

    return sequence


def cluster_sequences(list_sequences, seqid=1.0, coverage=0.8, covmode='0', path2mmseqstmp='/tmp', path2mmseqs='mmseqs'):
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


def split_csv(input_csv, output_dir, test_size=0.2, val_size=0.1, group_by_ligand=True, group_by_homology=False, base_data_path=LIGAND_DIR):

    # Read the CSV file
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    # df = df[:500]
    print(f"Read CSV with {len(df)} rows")
    df = df.drop_duplicates(subset=['ref_protein', 'mov_protein'], keep=False)
    print(f"Removed duplicates, {len(df)} rows remaining")

    for col in  ['bbr', 'bbc']:
        if df[col].apply(lambda x: isinstance(x, str) and x.startswith('[') and x.endswith(']')).any():
            df[col] = df[col].fillna('[]')
            
            df[col] = df[col].apply(lambda x: json.loads(x))
            df[col] = df[col].apply(lambda x: deserialize_nested_lists(x, col))
    
    df['bbr'] = df['bbr'].apply(lambda x: max([max(y) for y in x if len(y) > 0], default=float('-inf')))
    df['bbc'] = df['bbc'].apply(lambda x: max([max(y) for y in x if len(y) > 0], default=float('-inf')))
    # df = df[df['bbr'] > 0.3]
    df = df[df['bbc'] > 10]
    print(f"Filtered Best BBR and BBC, {len(df)} rows remaining")
    # df = df.groupby('ref_protein').head(50)
    # df = df.groupby('mov_protein').head(50)
    high_cath_degree = df[df['cath_degree'] > 3]
    print(f"Grouped by ref_protein and mov_protein, {len(df)} rows remaining")
    high_cath_degree = high_cath_degree.groupby('Ligand_ID').head(1000)
    df = pd.concat([df[df['cath_degree'] < 4], high_cath_degree])
    print(f"Grouped by Ligand_ID, {len(df)} rows remaining")

    # Ensure the "Ligand_ID" column exists if group_by_ligand is True
    if group_by_ligand and "Ligand_ID" not in df.columns:
        raise ValueError("The CSV file must contain a 'Ligand_ID' column when group_by_ligand is enabled")

    # Split based on Ligand_ID grouping if specified
    if group_by_ligand:
        # Get unique Ligand_IDs and shuffle
        ligand_ids = df['Ligand_ID'].unique()
        random.seed(42)  # For reproducibility
        random.shuffle(ligand_ids)

        # Calculate number of ligands for each split
        total_ligands = len(ligand_ids)
        test_len = int(total_ligands * test_size)
        val_len = int(total_ligands * val_size)

        # Assign Ligand_IDs to each split
        test_ligands = ligand_ids[:test_len]
        val_ligands = ligand_ids[test_len:test_len + val_len]
        train_ligands = ligand_ids[test_len + val_len:]

        # Create datasets for each split based on Ligand_ID
        train_df = df[df['Ligand_ID'].isin(train_ligands)]
        val_df = df[df['Ligand_ID'].isin(val_ligands)]
        val_df = val_df.groupby('Ligand_ID').head(300)  # Ensure each ligand is present in the validation set
        test_df = df[df['Ligand_ID'].isin(test_ligands)]
    
    elif group_by_homology:
        from Bio.PDB import PDBParser, PPBuilder
        import warnings
        from Bio.PDB.PDBExceptions import PDBConstructionWarning
        import subprocess

        warnings.simplefilter('ignore', PDBConstructionWarning)

        def extract_sequence_from_pdb(pdb_file):
            parser = PDBParser(QUIET=True)
            ppb = PPBuilder()

            structure = parser.get_structure("pdb", pdb_file)
            model = list(structure)[0]
            chain = list(model)[0]  # use first chain
            peptides = ppb.build_peptides(chain)

            if not peptides:
                raise ValueError(f"No peptides found in {pdb_file}")

            sequence = "".join(str(peptide.get_sequence()) for peptide in peptides)
            return sequence

        def cluster_sequences(list_sequences, seqid=1.0, coverage=0.8, covmode='0',
                              path2mmseqstmp='/tmp', path2mmseqs='/home/iscb/wolfson/hagairavid/miniforge3/envs/aligner_dl2/bin/mmseqs'):
            rng = np.random.randint(0, high=int(1e6))
            tmp_input = os.path.join(path2mmseqstmp, f'tmp_input_file_{rng}.fasta')
            tmp_output = os.path.join(path2mmseqstmp, f'tmp_output_file_{rng}')

            with open(tmp_input, 'w') as f:
                for k, sequence in enumerate(list_sequences):
                    f.write(f'>{k}\n{sequence}\n')

            command = f'{path2mmseqs} easy-cluster {tmp_input} {tmp_output} {path2mmseqstmp} --min-seq-id {seqid} -c {coverage} --cov-mode {covmode}'
            print(f'Running: {command}')
            subprocess.run(command.split(' '), check=True)

            rep_file = tmp_output + '_rep_seq.fasta'
            cluster_file = tmp_output + '_cluster.tsv'
            representative_indices = [int(x[1:-1]) for x in open(rep_file).readlines()[::2]]

            cluster_indices = np.zeros(len(list_sequences), dtype=int)
            table = pd.read_csv(cluster_file, sep='\t', header=None).to_numpy(dtype=int)
            for i, j in table:
                if i in representative_indices:
                    cluster_indices[j] = representative_indices.index(i)

            # Clean up
            for suffix in ['_rep_seq.fasta', '_all_seqs.fasta', '_cluster.tsv']:
                try:
                    os.remove(tmp_output + suffix)
                except FileNotFoundError:
                    pass
            os.remove(tmp_input)

            return cluster_indices

        if base_data_path is None:
            raise ValueError("base_data_path must be specified when using group_by_homology")

        # Extract unique protein sequences
        protein_to_sequence = {}
        protein_to_index = {}
        sequence_list = []

        all_proteins = pd.unique(df[['ref_protein', 'mov_protein']].values.ravel())
        # if os.path.exists(os.path.join(output_dir, 'protein_sequences.json')):
        #     with open(os.path.join(output_dir, 'protein_sequences.json'), 'r') as f:
        #         protein_to_sequence = json.load(f)
        #     sequence_list = list(protein_to_sequence.values())
        #     protein_to_index = {pid: idx for idx, pid in enumerate(protein_to_sequence.keys())}
        #     print(f"Loaded {len(sequence_list)} cached sequences")
        # else:
        print("No cached sequences found, extracting from PDB files")
        for protein_id in all_proteins:
            ligand_id_row = df.loc[df['ref_protein'] == protein_id]
            if ligand_id_row.empty:
                ligand_id_row = df.loc[df['mov_protein'] == protein_id]
            if ligand_id_row.empty:
                continue
            ligand_id = ligand_id_row.iloc[0]['Ligand_ID']

            pdb_file = os.path.join(base_data_path, ligand_id, protein_id + '_non_ligand.ent')
            try:
                seq = extract_sequence_from_pdb(pdb_file, chain_id = ligand_id_row)
                protein_to_sequence[protein_id] = seq
                protein_to_index[protein_id] = len(sequence_list)
                sequence_list.append(seq)
            except Exception as e:
                print(f"[Warning] Skipping {protein_id}: {e}")

        # cache sequences
        with open(os.path.join(output_dir, 'protein_sequences.json'), 'w') as f:
            json.dump(protein_to_sequence, f)
        print(f"Extracted {len(sequence_list)} sequences")

        # Cluster sequences
        cluster_ids = cluster_sequences(sequence_list, seqid=0.2, coverage=0.2)
        protein_to_cluster = {pid: cluster_ids[idx] for pid, idx in protein_to_index.items()}

        # Assign clusters to each row
        df['ref_cluster'] = df['ref_protein'].map(protein_to_cluster)
        df['mov_cluster'] = df['mov_protein'].map(protein_to_cluster)
        df = df.dropna(subset=['ref_cluster', 'mov_cluster'])

        # Map from cluster_id to list of proteins
        cluster_to_proteins = defaultdict(set)
        for pid, cluster_id in protein_to_cluster.items():
            cluster_to_proteins[cluster_id].add(pid)

        # Shuffle clusters
        cluster_ids_unique = list(cluster_to_proteins.keys())
        random.shuffle(cluster_ids_unique)

        # Assign clusters to splits
        total = len(cluster_ids_unique)
        n_test = int(test_size * total)
        n_val = int(val_size * total)

        test_clusters = set(cluster_ids_unique[:n_test])
        val_clusters = set(cluster_ids_unique[n_test:n_test + n_val])
        train_clusters = set(cluster_ids_unique[n_test + n_val:])

        # Assign proteins to sets
        protein_to_split = {}
        for cluster in train_clusters:
            for protein in cluster_to_proteins[cluster]:
                protein_to_split[protein] = 'train'
        for cluster in val_clusters:
            for protein in cluster_to_proteins[cluster]:
                protein_to_split[protein] = 'val'
        for cluster in test_clusters:
            for protein in cluster_to_proteins[cluster]:
                protein_to_split[protein] = 'test'

        # Only keep rows where both proteins belong to the same split
        split_dfs = {'train': [], 'val': [], 'test': []}
        for _, row in df.iterrows():
            ref_split = protein_to_split.get(row['ref_protein'])
            mov_split = protein_to_split.get(row['mov_protein'])
            if ref_split is not None and ref_split == mov_split:
                split_dfs[ref_split].append(row)

        train_df = pd.DataFrame(split_dfs['train'])
        val_df = pd.DataFrame(split_dfs['val'])
        test_df = pd.DataFrame(split_dfs['test'])

        print(f"Train clusters: {len(train_clusters)}, rows: {len(train_df)}")
        print(f"Val clusters: {len(val_clusters)}, rows: {len(val_df)}")
        print(f"Test clusters: {len(test_clusters)}, rows: {len(test_df)}")

    else :
        # Step 1: Get the unique set of all proteins
        all_proteins = set(df['ref_protein']).union(set(df['mov_protein']))
        all_proteins = list(all_proteins)
        random.seed(42)
        random.shuffle(all_proteins)

        # Step 2: Split protein set
        total = len(all_proteins)
        test_cutoff = int(total * test_size)
        val_cutoff = int(total * val_size)

        test_proteins = set(all_proteins[:test_cutoff])
        val_proteins = set(all_proteins[test_cutoff:test_cutoff + val_cutoff])
        train_proteins = set(all_proteins[test_cutoff + val_cutoff:])

        # Step 3: Assign rows only if both ref and mov proteins are from the same split
        train_df = df[df['ref_protein'].isin(train_proteins) & df['mov_protein'].isin(train_proteins)]
        val_df = df[df['ref_protein'].isin(val_proteins) & df['mov_protein'].isin(val_proteins)]
        test_df = df[df['ref_protein'].isin(test_proteins) & df['mov_protein'].isin(test_proteins)]

        print(f"Train proteins: {len(train_proteins)}, rows: {len(train_df)}")
        print(f"Val proteins: {len(val_proteins)}, rows: {len(val_df)}")
        print(f"Test proteins: {len(test_proteins)}, rows: {len(test_df)}")

    # Save the splits into CSV files
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"Data split completed. Files and histograms saved in {output_dir}")

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Split CSV by Ligand_ID column or regular row-based split")
    parser.add_argument('--input_csv', type=str, required=True, help="Path to the input CSV file")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output CSV files")
    parser.add_argument('--test_size', type=float, default=0.2, help="Fraction of the data to be used as the test set")
    parser.add_argument('--val_size', type=float, default=0.1, help="Fraction of the total data to be used as validation set")
    parser.add_argument('--group_by_ligand', action='store_true', help="Enable to group rows by Ligand_ID")
    parser.add_argument('--group_by_homology', action='store_true', help="Group proteins by sequence similarity using MMseqs2")


    args = parser.parse_args()

    # Call the function to split the CSV
    split_csv(args.input_csv, args.output_dir, test_size=args.test_size, val_size=args.val_size, group_by_ligand=args.group_by_ligand, group_by_homology=args.group_by_homology, base_data_path=LIGAND_DIR)
