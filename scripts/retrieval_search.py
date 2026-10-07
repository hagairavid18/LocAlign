"""Retrieval benchmark: run one LocAlign database search per query.

Each query (a row of datasets/retrieval/queries.csv) is a BioLiP chain together
with the ligand-binding motif of its src_ligand. In inference.py's
--protein_database_search mode the query plays the *target* role (tar_protein,
tar_chain, tar_motif, tar_ligand) and every database row plays the *source*
role (src_protein, src_chain, src_motif, src_ligand, src_ligand_n_atoms), so the
query motif is passed as tar_motif and defines the query pocket.

No pLRMSD threshold is applied to the ranking: inference_results.csv always
contains every database pair. The max_pLRMSD / max_pLRMSD_normed thresholds in
inference.py only gate the per-hit Chimera visualization folders, so we set
them to -inf to skip visualization entirely (it is not needed for retrieval).

Modes
  --precompute          populate the shared feature cache (structures, ScanNet,
                        ESM) for every database chain, without running the model.
  --query_index I       run the search for query I and write the ranked hit list
                        <hits_dir>/qIII.csv.gz (skipped if it already exists).

Seeds: random / numpy / torch are seeded with --seed (default 42) before
anything else; DataLoader worker seeds derive from the torch seed.

The calibration models (calibration_model.pkl, ligand_calibration_model.pkl) are
picked up from the checkpoint directory, as in inference.py. Hit lists are built
from the in-memory results (full float precision; inference_results.csv is
rounded to 3 decimals), with the query chain itself removed.

Usage (defaults: RETRIEVAL_DATABASE_CSV, RETRIEVAL_CHECKPOINT, RETRIEVAL_QUERIES_CSV,
RETRIEVAL_CLUSTERS_CSV, <work>/search, <work>/hits)
  python3 scripts/retrieval_search.py --precompute
  python3 scripts/retrieval_search.py --query_index 0
"""
import argparse
import hashlib
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'aligner_dl'))

from scripts.inference import InferenceRunner  # noqa: E402
from utils.constants import (  # noqa: E402
    RETRIEVAL_CHECKPOINT,
    RETRIEVAL_CLUSTERS_CSV,
    RETRIEVAL_DATABASE_CSV,
    RETRIEVAL_HIT_FILE_FORMAT,
    RETRIEVAL_HITS_DIR,
    RETRIEVAL_QUERIES_CSV,
    RETRIEVAL_SEARCH_DIR,
)

DEFAULTS = dict(
    database=RETRIEVAL_DATABASE_CSV,
    checkpoint=RETRIEVAL_CHECKPOINT,
    queries=RETRIEVAL_QUERIES_CSV,
    clusters=RETRIEVAL_CLUSTERS_CSV,
    base_save_dir=RETRIEVAL_SEARCH_DIR,
    hits_dir=RETRIEVAL_HITS_DIR,
)


def seed_everything(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def motif_to_cli(
    val,
) -> str:
    motif = InferenceRunner._parse_motif(val)
    if not motif:
        raise ValueError(f'Query has no motif: {val!r}')
    return ','.join(str(int(x)) for x in motif)


def make_runner(
    args,
    protein,
    chain,
    ligand,
    motif,
):
    return InferenceRunner(
        checkpoint_path=args.checkpoint,
        base_save_dir=args.base_save_dir,
        protein_database_search=(protein, chain, args.database),
        tar_motif=motif,
        tar_ligand_id=ligand,
        max_pLRMSD=-np.inf,
        max_pLRMSD_normed=-np.inf,
    )


def precompute(
    args,
) -> None:
    """Download structures, run ScanNet and compute ESM embeddings for every DB chain.

    A corrupt cached ESM embedding (unreadable .pt) is removed and recomputed once;
    the work cache is hard-linked from an older cache, so removing only unlinks this
    cache's name.
    """
    db = pd.read_csv(args.database, dtype=str)
    q = db.iloc[0]
    runner = make_runner(args, q.src_protein, q.src_chain, q.src_ligand, motif_to_cli(q.src_motif))
    runner._prepare_dataframe()
    runner._setup_directories()
    runner._save_non_ligand_models()
    runner._extract_features()
    runner._write_resrced_pairs()
    runner._prepare_dataloader()
    ds = runner._dataloader.dataset
    base = runner._cache_paths['pdb_files']
    todo = sorted({(p.src_protein, p.src_chain, p.src_ligand) for p in runner._pairs})
    n_fail, t0 = 0, time.time()
    for i, (prot, chain, lig) in enumerate(todo):
        pdb_file = os.path.join(base, lig, f'{prot}{chain}_non_ligand_.ent')
        try:
            ds.extract_esm_embeddings(pdb_file=pdb_file, chain_name=f'{prot}_{chain}')
        except Exception as e:  # noqa: BLE001
            cache_path = os.path.join(ds._cache_dir, hashlib.md5(os.path.basename(pdb_file).encode()).hexdigest() + '.pt')
            recovered = False
            if os.path.exists(cache_path) and os.path.exists(pdb_file):
                print(f'ESM cache unreadable for {prot}{chain} ({e}); recomputing')
                os.remove(cache_path)
                try:
                    ds.extract_esm_embeddings(pdb_file=pdb_file, chain_name=f'{prot}_{chain}')
                    recovered = True
                except Exception as e2:  # noqa: BLE001
                    e = e2
            if not recovered:
                n_fail += 1
                print(f'ESM failed for {prot}{chain} ({lig}): {e}')
        if i % 2000 == 0:
            print(f'ESM {i}/{len(todo)} ({time.time() - t0:.0f}s)', flush=True)
    print(f'Precompute done. {len(todo)} chains, ESM failures: {n_fail}')


def run_query(
    args,
) -> None:
    queries = pd.read_csv(args.queries, dtype=str)
    q = queries[queries['query_idx'].astype(int) == args.query_index].iloc[0]
    os.makedirs(args.hits_dir, exist_ok=True)
    out_path = os.path.join(args.hits_dir, RETRIEVAL_HIT_FILE_FORMAT.format(args.query_index))
    if os.path.exists(out_path) and not args.overwrite:
        print(f'{out_path} exists, skipping query {args.query_index}')
        return

    print(f'Query {args.query_index}: {q.src_protein}{q.src_chain} ligand={q.src_ligand} motif={q.src_motif}')
    runner = make_runner(args, q.src_protein, q.src_chain, q.src_ligand, motif_to_cli(q.src_motif))
    runner.run()

    res = pd.DataFrame([p.to_dict() for p in getattr(runner, '_all_pairs', runner._pairs)])
    res['pLRMSD_normalized'] = res['pLRMSD'].astype(float) / res['eLRMSD'].astype(float)
    res['chain_id'] = res['src_protein'] + res['src_chain']
    query_chain = f'{q.src_protein}{q.src_chain}'
    n_self = int((res['chain_id'] == query_chain).sum())
    res = res[res['chain_id'] != query_chain]

    clusters = pd.read_csv(args.clusters, dtype=str)
    cmap = dict(zip(clusters['chain_id'], clusters['cluster_id']))
    hits = pd.DataFrame({
        'chain_id': res['chain_id'],
        'cluster_id': res['chain_id'].map(cmap),
        'ligand': res['src_ligand'],
        'normalized_pLRMSD': res['pLRMSD_normalized'],
    }).sort_values('normalized_pLRMSD', ascending=True, na_position='last', kind='mergesort')
    tmp = out_path + '.tmp.csv.gz'
    hits.to_csv(tmp, index=False, float_format='%.6f')
    os.replace(tmp, out_path)
    n_nan = int(hits['normalized_pLRMSD'].isna().sum())
    print(f'Wrote {len(hits)} hits ({n_nan} without score, {n_self} self rows removed) to {out_path}')
    print(hits.head(10).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--precompute', action='store_true')
    g.add_argument('--query_index', type=int)
    for k, v in DEFAULTS.items():
        ap.add_argument(f'--{k}', default=v)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    seed_everything(args.seed)
    with torch.inference_mode():
        if args.precompute:
            precompute(args)
        else:
            run_query(args)


if __name__ == '__main__':
    main()
