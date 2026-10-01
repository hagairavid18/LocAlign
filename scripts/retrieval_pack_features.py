"""Retrieval benchmark: pack the per-chain DB features into a few large shard files.

scripts/inference.py reads, for every (query, DB entry) pair, one gzipped ScanNet
pickle and one ESM .pt file from the shared cache on NFS. Over 32.6k entries and
100 queries that is ~6.5M small-file reads, and NFS latency (not the GPU) becomes
the bottleneck. This script reads each chain's two files once and writes them
unchanged (same arrays, same dtypes; only keys the dataset never reads are
dropped) into ~33 uncompressed pickle shards, so a search job reads a few large
files sequentially instead.

Per chain the shard stores
  'scannet': the ScanNet feature dict restricted to the keys that
             ScanNetDataset._read_embedding uses
  'esm_keys', 'esm': the ESM cache dict {resseq: tensor(640)} as an int64 key
             array and a float32 matrix (identical values)
Output: <out>/shard_XXX.pkl and <out>/manifest.csv (chain_id, shard, status).
Resumable: existing shards are kept; each shard is written to a temp file and renamed.
"""
import argparse
import gzip
import hashlib
import os
import pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

MAIN_REPO = '/home/iscb/wolfson/hagairavid/LocAlign'
WORK = '/home/iscb/wolfson/hagairavid/LocAlign_retrieval_work'
SCANNET_KEYS = ('sequence_indices_atom', 'atomic_embeddings', 'atomic_plus_residue_embedding',
                'residue_embeddings', 'residue_ids', 'atomic_frames', 'atom_nearest_neighbors', 'atom_types')
ESM_SUBDIR = 'esm_embeddings/esm_cache_esm2_t30_150M_UR50D_18'


def load_chain(cache, chain_id):
    sc_path = os.path.join(cache, 'scannet_embeddings', f'{chain_id}_scannet_atoms.pkl')
    esm_path = os.path.join(cache, ESM_SUBDIR,
                            hashlib.md5(f'{chain_id}_non_ligand_.ent'.encode()).hexdigest() + '.pt')
    if not os.path.exists(sc_path):
        return None, 'no_scannet'
    if not os.path.exists(esm_path):
        return None, 'no_esm'
    try:
        with gzip.open(sc_path, 'rb') as f:
            data = pickle.load(f)
        esm = torch.load(esm_path)
    except Exception as e:  # noqa: BLE001
        return None, f'unreadable:{type(e).__name__}'
    keys = np.array(sorted(esm.keys()), dtype=np.int64)
    mat = np.stack([esm[int(k)].numpy() for k in keys]).astype(np.float32, copy=False) if len(keys) \
        else np.zeros((0, 640), np.float32)
    sc = {k: data[k] for k in SCANNET_KEYS if k in data}
    return {'scannet': sc, 'esm_keys': keys, 'esm': mat}, 'ok'


def pack_shard(job):
    shard_idx, chains, cache, out = job
    path = os.path.join(out, f'shard_{shard_idx:03d}.pkl')
    status = []
    store = {}
    for c in chains:
        item, st = load_chain(cache, c)
        status.append((c, shard_idx, st))
        if item is not None:
            store[c] = item
    if not os.path.exists(path):
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(store, f, protocol=5)
        os.replace(tmp, path)
    print(f'shard {shard_idx}: {len(store)}/{len(chains)} chains', flush=True)
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--database', default=f'{MAIN_REPO}/example_inputs/biolip2_nr_database_with_motif.csv')
    ap.add_argument('--cache', default=f'{WORK}/search/.cache')
    ap.add_argument('--out', default=f'{WORK}/packed_features')
    ap.add_argument('--shard_size', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()

    db = pd.read_csv(args.database, dtype=str)
    chains = sorted(set(db['src_protein'] + db['src_chain']))
    os.makedirs(args.out, exist_ok=True)
    jobs = [(i, chains[s:s + args.shard_size], args.cache, args.out)
            for i, s in enumerate(range(0, len(chains), args.shard_size))]
    with Pool(args.workers) as pool:
        status = [row for rows in pool.imap_unordered(pack_shard, jobs) for row in rows]
    man = pd.DataFrame(status, columns=['chain_id', 'shard', 'status']).sort_values(['shard', 'chain_id'])
    man.to_csv(os.path.join(args.out, 'manifest.csv'), index=False)
    print(man['status'].value_counts().to_string())


if __name__ == '__main__':
    main()
