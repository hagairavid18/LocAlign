"""Retrieval benchmark (rebuttal R1 C7): database clustering and query selection.

Subcommands
  link-chains     Make <work>/chains/<chain_id>.pdb symlinks to the single-chain
                  structures that inference.py caches for each DB row
                  (<cache>/pdb_files/<src_ligand>/<pdb><chain>_non_ligand_.ent,
                  i.e. the chain with the entry's own ligand removed). These are
                  the Foldseek input. Chains without a cached structure are
                  written to <work>/chains_missing.csv.
  clusters-csv    Convert Foldseek's <prefix>_cluster.tsv into
                  datasets/retrieval/foldseek_clusters_tm06.csv
                  (chain_id, cluster_id, cluster_size); cluster_id is the
                  Foldseek representative chain.
  select-queries  Pick the retrieval queries (seed 42) and write
                  datasets/retrieval/queries.csv.

Definitions (used by select-queries and scripts/retrieval_metrics.py)
  entry    one row of example_inputs/biolip2_nr_database_with_motif.csv
  chain_id src_protein + src_chain (e.g. "12asA")
  ligand   the entry's src_ligand (the ligand whose binding residues form
           src_motif). ligand_all is NOT used.
"""
import argparse
import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_REPO = '/home/iscb/wolfson/hagairavid/LocAlign'
WORK = '/home/iscb/wolfson/hagairavid/LocAlign_retrieval_work'
DB = f'{MAIN_REPO}/example_inputs/biolip2_nr_database_with_motif.csv'
CLUSTERS = os.path.join(REPO_ROOT, 'datasets/retrieval/foldseek_clusters_tm06.csv')
QUERIES = os.path.join(REPO_ROOT, 'datasets/retrieval/queries.csv')


def read_db(path=DB):
    db = pd.read_csv(path, dtype=str)
    db['chain_id'] = db['src_protein'] + db['src_chain']
    return db


def link_chains(args):
    db = read_db()
    out = os.path.join(args.work, 'chains')
    os.makedirs(out, exist_ok=True)
    missing, n = [], 0
    for r in db.drop_duplicates('chain_id').itertuples():
        src = os.path.join(args.cache, 'pdb_files', r.src_ligand, f'{r.src_protein}{r.src_chain}_non_ligand_.ent')
        dst = os.path.join(out, f'{r.chain_id}.pdb')
        if os.path.exists(src):
            if not os.path.lexists(dst):
                os.symlink(src, dst)
            n += 1
        else:
            missing.append((r.chain_id, r.src_protein, r.src_chain, r.src_ligand))
    pd.DataFrame(missing, columns=['chain_id', 'src_protein', 'src_chain', 'src_ligand']).to_csv(
        os.path.join(args.work, 'chains_missing.csv'), index=False)
    print(f'{db.chain_id.nunique()} unique chains in {len(db)} DB rows; linked {n}; missing {len(missing)}')


def _strip(name):
    name = name.split('.pdb')[0] if '.pdb' in name else name
    return name


def clusters_csv(args):
    tsv = pd.read_csv(args.tsv, sep='\t', header=None, names=['rep', 'member'], dtype=str)
    tsv['cluster_id'] = tsv['rep'].map(_strip)
    tsv['chain_id'] = tsv['member'].map(_strip)
    out = tsv[['chain_id', 'cluster_id']].drop_duplicates('chain_id')
    out['cluster_size'] = out.groupby('cluster_id')['chain_id'].transform('size')
    out = out.sort_values(['cluster_id', 'chain_id'])
    os.makedirs(os.path.dirname(CLUSTERS), exist_ok=True)
    out.to_csv(CLUSTERS, index=False)
    sizes = out.drop_duplicates('cluster_id')['cluster_size']
    print(f'{len(out)} chains, {len(sizes)} clusters, {(sizes == 1).sum()} singletons, '
          f'largest {sizes.max()}')


def select_queries(args):
    rng = np.random.RandomState(args.seed)
    db = read_db()
    cl = pd.read_csv(CLUSTERS, dtype={'chain_id': str, 'cluster_id': str})
    db = db.merge(cl, on='chain_id', how='left')
    n0 = len(db)
    n_unclustered = int(db['cluster_id'].isna().sum())
    # a) drop entries without a cluster (no structure) or in singleton clusters
    db = db[db['cluster_id'].notna() & (db['cluster_size'] > 1)]
    n_a = len(db)
    # b) ligand counts over the remaining entries: #entries and #distinct clusters
    stats = db.groupby('src_ligand').agg(n_entries=('chain_id', 'size'),
                                         n_clusters=('cluster_id', 'nunique'))
    eligible = stats[(stats.n_entries >= 5) & (stats.n_clusters >= 3)].sort_index()
    # c) one seeded-random entry per ligand; then sample N ligands if more remain
    ligands = list(eligible.index)
    if len(ligands) > args.n:
        ligands = sorted(rng.choice(ligands, size=args.n, replace=False).tolist())
    rows = []
    for lig in ligands:
        cand = db[db['src_ligand'] == lig].sort_values(['chain_id']).reset_index(drop=True)
        rows.append(cand.iloc[rng.randint(len(cand))])
    q = pd.DataFrame(rows).reset_index(drop=True)
    q = q.merge(eligible, left_on='src_ligand', right_index=True)
    q['cluster_size'] = q['cluster_size'].astype(int)
    q.insert(0, 'query_idx', range(len(q)))
    cols = ['query_idx', 'chain_id', 'src_protein', 'src_chain', 'src_ligand', 'src_motif',
            'src_ligand_n_atoms', 'cluster_id', 'cluster_size', 'n_entries', 'n_clusters']
    q[cols].to_csv(QUERIES, index=False)
    print(f'DB rows {n0}; without cluster {n_unclustered}; after dropping singleton/unclustered {n_a}; '
          f'ligands in remaining {len(stats)}; eligible ligands (>=5 entries, >=3 clusters) {len(eligible)}; '
          f'queries {len(q)} (seed {args.seed})')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('link-chains')
    p.add_argument('--cache', default=f'{WORK}/search/.cache')
    p.add_argument('--work', default=WORK)
    p = sub.add_parser('clusters-csv')
    p.add_argument('--tsv', default=f'{WORK}/foldseek/clu_tm06_cluster.tsv')
    p = sub.add_parser('select-queries')
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    {'link-chains': link_chains, 'clusters-csv': clusters_csv, 'select-queries': select_queries}[args.cmd](args)


if __name__ == '__main__':
    main()
