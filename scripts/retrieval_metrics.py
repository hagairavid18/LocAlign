"""Retrieval benchmark (rebuttal R1 C7): metrics from the per-query ranked hit lists.

Input hit lists are sorted by normalized_pLRMSD ascending, with the query chain
already removed (scripts/retrieval_search.py). Hits without a score (pairs that
failed feature loading) are ignored. A hit "has the same ligand" if its
src_ligand equals the query's src_ligand; it is "in the query's cluster" if its
Foldseek (TM >= 0.6) cluster_id equals the query's cluster_id (chains with no
cluster never match).

  Top_K_retrieval               fraction of queries with >= 1 of the top-K hits
                                having the same ligand OR the same cluster.
  Top_K_retrieval_not_FoldSeek  drop all hits in the query's cluster; fraction of
                                queries with >= 1 of the top-K remaining hits
                                having the same ligand.

Usage
  # from the full per-query lists (<work>/hits/qIII.csv.gz), also exporting the
  # compact committed list results/retrieval/hits_top200.csv.gz
  python scripts/retrieval_metrics.py --hits_dir /home/iscb/wolfson/hagairavid/LocAlign_retrieval_work/hits --export
  # recompute from the committed compact list
  python scripts/retrieval_metrics.py
The compact list keeps, per query, the top-200 hits overall plus the top-200 hits
outside the query's cluster, so every metric here is exactly reproducible from it.
"""
import argparse
import glob
import os

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUERIES = os.path.join(REPO_ROOT, 'datasets/retrieval/queries.csv')
OUT_DIR = os.path.join(REPO_ROOT, 'results/retrieval')
COMPACT = os.path.join(OUT_DIR, 'hits_top200.csv.gz')
KS = (1, 3, 5, 10)
KEEP = 200


def load_full(hits_dir, queries):
    frames = []
    for path in sorted(glob.glob(os.path.join(hits_dir, 'q[0-9][0-9][0-9].csv.gz'))):
        qi = int(os.path.basename(path)[1:4])
        h = pd.read_csv(path, dtype={'chain_id': str, 'cluster_id': str, 'ligand': str})
        h = h[h['normalized_pLRMSD'].notna()].reset_index(drop=True)
        h.insert(0, 'rank', range(1, len(h) + 1))
        h.insert(0, 'query_idx', qi)
        frames.append(h)
    return pd.concat(frames, ignore_index=True)


def annotate(hits, queries):
    q = queries.set_index('query_idx')
    hits = hits.copy()
    hits['same_ligand'] = hits['ligand'].values == q.loc[hits['query_idx'], 'src_ligand'].values
    hits['same_cluster'] = (hits['cluster_id'].values == q.loc[hits['query_idx'], 'cluster_id'].values) \
        & hits['cluster_id'].notna().values
    return hits


def compact(hits):
    keep = []
    for _, h in hits.groupby('query_idx', sort=True):
        h = h.sort_values('rank')
        keep.append(pd.concat([h.head(KEEP), h[~h['same_cluster']].head(KEEP)]).drop_duplicates('rank'))
    return pd.concat(keep).sort_values(['query_idx', 'rank'])


def metrics(hits, queries):
    per_q = []
    for qi in queries['query_idx']:
        h = hits[hits['query_idx'] == qi].sort_values('rank')
        row = {'query_idx': qi, 'n_hits_scored': len(h)}
        nc = h[~h['same_cluster']]
        for k in KS:
            top, top_nc = h.head(k), nc.head(k)
            row[f'top{k}_hit'] = bool((top['same_ligand'] | top['same_cluster']).any())
            row[f'top{k}_hit_not_foldseek'] = bool(top_nc['same_ligand'].any())
        lig = h[h['same_ligand']]
        row['first_same_ligand_rank'] = int(lig['rank'].iloc[0]) if len(lig) else None
        lig_nc = nc[nc['same_ligand']]
        row['first_same_ligand_rank_not_foldseek'] = int(lig_nc['rank'].iloc[0]) if len(lig_nc) else None
        per_q.append(row)
    per_q = pd.DataFrame(per_q)
    rows = []
    for k in KS:
        rows.append({'K': k,
                     'Top_K_retrieval': per_q[f'top{k}_hit'].mean(),
                     'Top_K_retrieval_not_FoldSeek': per_q[f'top{k}_hit_not_foldseek'].mean(),
                     'n_queries': len(per_q)})
    return pd.DataFrame(rows), per_q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hits_dir', default=None, help='full per-query lists; default: use the committed compact list')
    ap.add_argument('--export', action='store_true', help='write results/retrieval/hits_top200.csv.gz')
    ap.add_argument('--queries', default=QUERIES)
    args = ap.parse_args()

    queries = pd.read_csv(args.queries, dtype={'src_ligand': str, 'cluster_id': str, 'chain_id': str})
    if args.hits_dir:
        hits = annotate(load_full(args.hits_dir, queries), queries)
    else:
        hits = annotate(pd.read_csv(COMPACT, dtype={'chain_id': str, 'cluster_id': str, 'ligand': str}), queries)
    missing = sorted(set(queries['query_idx']) - set(hits['query_idx']))
    if missing:
        print(f'WARNING: no hit list for queries {missing}; they are excluded')
        queries = queries[~queries['query_idx'].isin(missing)]

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.export:
        c = compact(hits)
        c[['query_idx', 'rank', 'chain_id', 'cluster_id', 'ligand', 'normalized_pLRMSD']].to_csv(
            COMPACT, index=False, float_format='%.5f')
        print(f'Wrote {len(c)} rows to {COMPACT}')

    m, per_q = metrics(hits, queries)
    m.to_csv(os.path.join(OUT_DIR, 'metrics.csv'), index=False, float_format='%.4f')
    per_q.merge(queries[['query_idx', 'chain_id', 'src_ligand', 'cluster_id', 'cluster_size']],
                on='query_idx').to_csv(os.path.join(OUT_DIR, 'per_query.csv'), index=False)
    print(m.to_string(index=False))


if __name__ == '__main__':
    main()
