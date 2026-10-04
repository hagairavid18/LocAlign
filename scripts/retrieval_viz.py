"""ChimeraX scripts for the top retrieval hits of each query.

For each query, the top hits of its ranked hit list (scripts/retrieval_fast.py) are re-run
through InferenceRunner in pair mode (query as target, hit as source, both with their
motifs), which writes the predicted superposition and the aligned structures of each pair.
Outputs, per query, in <out_dir>/qIII/:
  chimera_script_multiple.cxc  the query receptor and ligand with the ligands of all
                               selected hits superposed, colored by rank
  <pair folder>/chimera_script_*.cxc  one hit: both receptors, both ligands and the
                               residue correspondences
  hits.csv                     the selected hits: retrieval rank, set (top / outside the
                               query's Foldseek cluster), same ligand, retrieval score and
                               the re-run scores, and the pair folder
Hits are the top --top overall plus the top --top_outside_cluster outside the query's
cluster, from the per-query hit lists (--hits_dir) or from the final ranked list exported by
scripts/retrieval_metrics.py --export (--hits_compact, e.g. after --patch_hits_dir).
Run on a GPU: CPU inference gives different scores for some pairs, so the drawn alignment
would not be the one that was ranked. A hit whose (chain, ligand) matches several database entries (3 such cases in
the database) is run once per entry. Re-run scores can differ slightly from the
retrieval scores for chains with more than 5000 atoms, which are subsampled at random
here (see scripts/retrieval_fast.py).

Usage
  python3 scripts/retrieval_viz.py --query_indices 0,1 --top 5 --top_outside_cluster 5
  # then, in ChimeraX: open <out_dir>/q000/chimera_script_multiple.cxc
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'aligner_dl'))
sys.path.insert(0, REPO_ROOT)

from scripts.chimera_pocket_viz import make_chimera_script_multiple  # noqa: E402
from scripts.inference import InferenceRunner  # noqa: E402
from utils.constants import (  # noqa: E402
    RETRIEVAL_CHECKPOINT,
    RETRIEVAL_DATABASE_CSV,
    RETRIEVAL_HIT_FILE_FORMAT,
    RETRIEVAL_HIT_FILE_PATTERN,
    RETRIEVAL_HITS_FAST_DIR,
    RETRIEVAL_QUERIES_CSV,
    RETRIEVAL_SEARCH_DIR,
    RETRIEVAL_VIZ_DIR,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--query_indices', default=None, help='comma-separated; default: every query with a hit list')
    ap.add_argument('--top', type=int, default=5, help='top hits overall')
    ap.add_argument('--top_outside_cluster', type=int, default=5, help="top hits outside the query's Foldseek cluster")
    ap.add_argument('--hits_dir', default=RETRIEVAL_HITS_FAST_DIR)
    ap.add_argument('--hits_compact', default=None,
                    help='ranked hit list exported by retrieval_metrics.py --export; used instead of --hits_dir')
    ap.add_argument('--queries', default=RETRIEVAL_QUERIES_CSV)
    ap.add_argument('--database', default=RETRIEVAL_DATABASE_CSV)
    ap.add_argument('--checkpoint', default=RETRIEVAL_CHECKPOINT)
    ap.add_argument('--cache_root', default=RETRIEVAL_SEARCH_DIR,
                    help='InferenceRunner base_save_dir; its .cache holds the structures and features')
    ap.add_argument('--out_dir', default=RETRIEVAL_VIZ_DIR)
    ap.add_argument('--overwrite', action='store_true', help='redo queries that already have an output folder')
    args = ap.parse_args()

    queries = pd.read_csv(args.queries, dtype=str)
    queries['query_idx'] = queries['query_idx'].astype(int)
    compact = None
    if args.hits_compact:
        compact = pd.read_csv(args.hits_compact, dtype={'chain_id': str, 'cluster_id': str, 'ligand': str})
    if args.query_indices:
        qidx = [int(x) for x in args.query_indices.split(',')]
    elif compact is not None:
        qidx = sorted(compact['query_idx'].unique())
    else:
        qidx = sorted(int(os.path.basename(p)[1:4]) for p in glob.glob(os.path.join(args.hits_dir, RETRIEVAL_HIT_FILE_PATTERN)))
    db = pd.read_csv(args.database, dtype=str)
    db['chain_id'] = db['src_protein'] + db['src_chain']
    os.makedirs(args.out_dir, exist_ok=True)

    for qi in qidx:
        q_dir = os.path.join(args.out_dir, f'q{qi:03d}')
        if os.path.exists(q_dir) and not args.overwrite:
            print(f'query {qi}: {q_dir} exists, skipped')
            continue
        if compact is not None:
            ranked = compact[compact['query_idx'] == qi].drop(columns='query_idx').sort_values('rank')
        else:
            hits_path = os.path.join(args.hits_dir, RETRIEVAL_HIT_FILE_FORMAT.format(qi))
            ranked = pd.read_csv(hits_path, dtype={'chain_id': str, 'cluster_id': str, 'ligand': str}) \
                if os.path.exists(hits_path) else pd.DataFrame()
            if len(ranked):
                ranked = ranked[ranked['normalized_pLRMSD'].notna()].reset_index(drop=True)
                ranked.insert(0, 'rank', range(1, len(ranked) + 1))
        if not len(ranked):
            print(f'query {qi}: no hit list, skipped')
            continue
        q = queries[queries['query_idx'] == qi].iloc[0]
        selected = select_hits(
            ranked,
            q,
            args.top,
            args.top_outside_cluster,
        )
        pairs = build_pairs(
            selected,
            q,
            db,
        )
        visualize_query(
            pairs,
            q_dir,
            args.checkpoint,
            args.cache_root,
        )


def select_hits(
    ranked: pd.DataFrame,
    query: pd.Series,
    top: int,
    top_outside_cluster: int,
) -> pd.DataFrame:
    """Top `top` hits plus top `top_outside_cluster` hits outside the query's cluster, by rank."""
    h = ranked.copy()
    h['same_ligand'] = h['ligand'] == query['src_ligand']
    h['in_query_cluster'] = h['cluster_id'] == query['cluster_id']
    top_all = h.head(top).assign(hit_set='top')
    top_out = h[~h['in_query_cluster']].head(top_outside_cluster).assign(hit_set='outside_cluster')
    both = pd.concat([top_all, top_out])
    both['hit_set'] = both.groupby('rank')['hit_set'].transform(lambda s: '+'.join(sorted(set(s))))
    return both.drop_duplicates('rank').sort_values('rank').reset_index(drop=True)


def build_pairs(
    selected: pd.DataFrame,
    query: pd.Series,
    db: pd.DataFrame,
) -> pd.DataFrame:
    """One InferenceRunner pair per selected hit: the query as target, the hit's database entry as source."""
    cols = ['chain_id', 'src_protein', 'src_chain', 'src_ligand', 'src_motif', 'src_ligand_n_atoms']
    pairs = selected.merge(db[cols], left_on=['chain_id', 'ligand'], right_on=['chain_id', 'src_ligand'], how='left')
    pairs['ambiguous_entry'] = pairs.duplicated('rank', keep=False)
    pairs = pairs.assign(
        tar_protein=query['src_protein'],
        tar_chain=query['src_chain'],
        tar_motif=query['src_motif'],
        tar_ligand=query['src_ligand'],
        tar_ligand_n_atoms=query['src_ligand_n_atoms'],
    )
    return pairs[pairs['src_protein'].notna()].reset_index(drop=True)


def visualize_query(
    pairs: pd.DataFrame,
    q_dir: str,
    checkpoint: str,
    cache_root: str,
) -> None:
    """Run the pairs through InferenceRunner and write the per-query ChimeraX script and hits.csv into q_dir.

    The runner writes into a new folder under `cache_root` (next to the feature cache it
    reuses); that folder is moved to q_dir when done.
    """
    os.makedirs(cache_root, exist_ok=True)
    pairs_csv = os.path.join(cache_root, f'viz_pairs_{os.path.basename(q_dir)}.csv')
    pairs.to_csv(pairs_csv, index=False)
    runner = InferenceRunner(
        checkpoint_path=checkpoint,
        base_save_dir=cache_root,
        csv_path=pairs_csv,
    )
    runner.run()
    rows = [ph.to_dict() for ph in runner._pairs]
    rerun = pd.DataFrame(rows)[['src_protein', 'src_chain', 'src_ligand', 'src_motif', 'pLRMSD', 'eLRMSD', 'output_folder']] \
        .rename(columns={'pLRMSD': 'rerun_pLRMSD', 'eLRMSD': 'rerun_eLRMSD'})
    rerun['src_motif'] = rerun['src_motif'].astype(str)
    pairs = pairs.drop(columns=[c for c in ('pLRMSD', 'eLRMSD') if c in pairs.columns])
    pairs = pairs.assign(src_motif=pairs['src_motif'].map(lambda m: str(InferenceRunner._parse_motif(m))))
    table = pairs.merge(rerun, on=['src_protein', 'src_chain', 'src_ligand', 'src_motif'], how='left')
    table['rerun_normalized_pLRMSD'] = table['rerun_pLRMSD'].astype(float) / table['rerun_eLRMSD'].astype(float)
    table['pLRMSD_normalized'] = table['rerun_normalized_pLRMSD']
    table = table.sort_values('rank', kind='mergesort').reset_index(drop=True)
    make_chimera_script_multiple(runner._output_dir, table, top=len(table))
    table['output_folder'] = table['output_folder'].map(lambda p: os.path.basename(p) if isinstance(p, str) else np.nan)
    table[['rank', 'hit_set', 'chain_id', 'cluster_id', 'ligand', 'same_ligand', 'in_query_cluster',
           'ambiguous_entry', 'normalized_pLRMSD', 'rerun_normalized_pLRMSD', 'rerun_pLRMSD', 'rerun_eLRMSD',
           'output_folder']].to_csv(os.path.join(runner._output_dir, 'hits.csv'), index=False, float_format='%.4f')
    if os.path.exists(q_dir):
        shutil.rmtree(q_dir)
    shutil.move(runner._output_dir, q_dir)
    os.remove(pairs_csv)
    print(f'{q_dir}: {len(table)} hits')


if __name__ == '__main__':
    main()
