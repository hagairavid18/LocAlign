"""Retrieval benchmark: fast multi-query database search from packed features.

Computes exactly what scripts/retrieval_search.py computes (one
--protein_database_search per query; query = target with its source motif, every
DB entry = source with its own motif and ligand size; score = pLRMSD / eLRMSD),
but avoids the NFS small-file bottleneck:

  * DB features come from the shards written by scripts/retrieval_pack_features.py,
    read sequentially one shard (~1000 chains) at a time into RAM;
  * several queries are processed per job, so each shard is read once per job
    and the model/ESM/dataset setup is paid once per job;
  * the per-file existence checks / download / ScanNet steps of inference.py are
    skipped; which chains have features is taken from the packing manifest
    (pairs whose chains have no features get NaN, as in inference.py).

The per-pair computation is unchanged: the same ScanNetDataset.__getitem__ (only
its two file loaders are redirected to the in-memory shard), the same model
inference_step, and InferenceRunner._process_visualization for pLRMSD/eLRMSD
(visualization disabled). Chains with more than max_length (5000) atoms are
randomly subsampled by the dataset, as in inference.py, so their scores are only
reproducible for a fixed seed and data order.

Resumable and preemption-safe: finished queries (<hits_dir>/qIII.csv.gz) are
skipped; per-shard partial scores are written atomically to
<partials_dir>/<group-hash>/shard_XXX.csv.gz and reused after a restart, so a
preempted job resumes at the next unfinished shard.

Usage: retrieval_fast.py --query_indices 0,1  |  --group G --group_size Q
"""
import argparse
import hashlib
import os
import pickle
import random
import sys
import time
import zlib

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from concurrent.futures import ThreadPoolExecutor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'aligner_dl'))

from scripts.inference import InferenceRunner  # noqa: E402
from scripts.retrieval_pack_features import load_chain  # noqa: E402
from scripts.retrieval_stage import ShardStager  # noqa: E402
from aligner_dl.datasets import ScanNetDataset  # noqa: E402
from aligner_dl.models.utils.collate import custom_collate_fn  # noqa: E402
from miners.utils.constants import PairHolder  # noqa: E402

MAIN_REPO = '/home/iscb/wolfson/hagairavid/LocAlign'
WORK = '/home/iscb/wolfson/hagairavid/LocAlign_retrieval_work'


class PackedScanNetDataset(ScanNetDataset):
    """ScanNetDataset whose feature loaders read from an in-memory store."""
    STORE: dict = {}
    # Chains with > max_length (5000) atoms are randomly subsampled inside
    # _read_embedding with the global numpy RNG, so in inference.py their scores
    # depend on batch/worker/data order (5.5% of the DB). With this flag the
    # subsample is seeded by the chain name instead: same atoms in every run,
    # every query and any batch size. Other chains are unaffected.
    DETERMINISTIC_SUBSAMPLE = True

    def _read_embedding(self, chain, ligand_id, esm_embedding_dict=None):
        if not self.DETERMINISTIC_SUBSAMPLE:
            return super()._read_embedding(chain=chain, ligand_id=ligand_id, esm_embedding_dict=esm_embedding_dict)
        state = np.random.get_state()
        np.random.seed(zlib.crc32(chain.encode()))
        try:
            return super()._read_embedding(chain=chain, ligand_id=ligand_id, esm_embedding_dict=esm_embedding_dict)
        finally:
            np.random.set_state(state)

    def _load_scannet_data(self, chain):
        item = self.STORE.get(chain)
        if item is None:
            raise ValueError(f'no packed features for {chain}')
        return item['scannet']

    def extract_esm_embeddings(self, pdb_file, chain_name):
        chain = os.path.basename(pdb_file)[:-len('_non_ligand_.ent')]
        item = self.STORE.get(chain)
        if item is None:
            raise ValueError(f'no packed features for {chain}')
        return {int(k): torch.from_numpy(item['esm'][i]) for i, k in enumerate(item['esm_keys'])}

    def __getitem__(self, idx):
        # inference.py's dataset silently substitutes a random other pair on error;
        # here every pair was pre-filtered to have features, so fail loudly instead.
        row = self._df.iloc[idx]
        for key in ('tar', 'src'):
            if row[f'{key}_protein'] + row[f'{key}_chain'] not in self.STORE:
                raise RuntimeError(f'missing features for pair {idx}')
        return super().__getitem__(idx)


def validate_packed(packed, max_atoms=5000):
    """Run ScanNetDataset._read_embedding on every packed chain (CPU) and mark
    chains that raise as 'read_error' in manifest.csv. inference.py would silently
    replace such pairs by a random other pair; we give them NaN instead."""
    ds = PackedScanNetDataset.__new__(PackedScanNetDataset)
    ds._max_atoms, ds._with_esm = max_atoms, True
    man_path = os.path.join(packed, 'manifest.csv')
    man = pd.read_csv(man_path, dtype={'chain_id': str})
    n_atoms = {}
    for si in sorted(man['shard'].unique()):
        with open(os.path.join(packed, f'shard_{si:03d}.pkl'), 'rb') as f:
            PackedScanNetDataset.STORE = pickle.load(f)
        for c in list(PackedScanNetDataset.STORE):
            try:
                esm = ds.extract_esm_embeddings(f'/x/{c}_non_ligand_.ent', c)
                emb = ds._read_embedding(chain=c, ligand_id=None, esm_embedding_dict=esm)
                n_atoms[c] = len(PackedScanNetDataset.STORE[c]['scannet']['atom_types'])
                assert emb['atom_embeddings'].shape[0] > 0
            except Exception as e:  # noqa: BLE001
                print(f'read_error {c}: {e!r}')
                man.loc[man['chain_id'] == c, 'status'] = 'read_error'
        print(f'validated shard {si}', flush=True)
    man['n_atoms'] = man['chain_id'].map(n_atoms)
    tmp = man_path + '.tmp'
    man.to_csv(tmp, index=False)
    os.replace(tmp, man_path)
    print(man['status'].value_counts().to_string())
    print(f'chains with > {max_atoms} atoms (randomly subsampled by the dataset): {int((man.n_atoms > max_atoms).sum())}')


def fp16_roundtrip(store):
    for item in store.values():
        item['esm'] = item['esm'].astype(np.float16).astype(np.float32)
        for k, v in item['scannet'].items():
            if isinstance(v, np.ndarray) and v.dtype == np.float32:
                item['scannet'][k] = v.astype(np.float16).astype(np.float32)


def load_shard(stager, si):
    # (Reading the whole file first and then pickle.loads was ~2x slower on a CPU node.)
    with open(stager.path(si), 'rb') as f:
        return pickle.load(f)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_csv(df, path):
    tmp = path + '.tmp.csv.gz'
    df.to_csv(tmp, index=False, float_format='%.6f')
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--query_indices')
    g.add_argument('--group', type=int)
    g.add_argument('--validate_packed', action='store_true')
    ap.add_argument('--group_size', type=int, default=10)
    ap.add_argument('--database', default=f'{MAIN_REPO}/example_inputs/biolip2_nr_database_with_motif.csv')
    ap.add_argument('--checkpoint', default=f'{MAIN_REPO}/checkpoints/baseline/epoch=9-step=87120.ckpt')
    ap.add_argument('--queries', default=os.path.join(REPO_ROOT, 'datasets/retrieval/queries.csv'))
    ap.add_argument('--clusters', default=os.path.join(REPO_ROOT, 'datasets/retrieval/foldseek_clusters_tm06.csv'))
    ap.add_argument('--packed', default=f'{WORK}/packed_features')
    ap.add_argument('--cache', default=f'{WORK}/search/.cache', help='per-file cache (only for query chains)')
    ap.add_argument('--hits_dir', default=f'{WORK}/hits_fast')
    ap.add_argument('--partials_dir', default=f'{WORK}/fast_partials')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--max_shards', type=int, default=None, help='debug: only the first N shards')
    ap.add_argument('--device', default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--stage_root', default=f"/tmp/localign_retrieval_{os.environ.get('USER', 'user')}",
                    help='node-local directory for a verified copy of the shards (see retrieval_stage.py)')
    ap.add_argument('--no_stage', action='store_true', help='always read shards from NFS')
    ap.add_argument('--stage_streams', type=int, default=8)
    ap.add_argument('--fp16_roundtrip', action='store_true',
                    help='evaluation only: round float32 features through float16 (simulates fp16 shards)')
    ap.add_argument('--random_subsample', action='store_true',
                    help='subsample >5000-atom chains with the global RNG, as inference.py does')
    args = ap.parse_args()
    PackedScanNetDataset.DETERMINISTIC_SUBSAMPLE = not args.random_subsample
    seed_everything(args.seed)
    t0 = time.time()
    if args.validate_packed:
        validate_packed(args.packed)
        return

    queries = pd.read_csv(args.queries, dtype=str)
    queries['query_idx'] = queries['query_idx'].astype(int)
    if args.query_indices is not None:
        qidx = [int(x) for x in args.query_indices.split(',')]
    else:
        qidx = list(range(args.group * args.group_size, (args.group + 1) * args.group_size))
    qidx = [i for i in qidx if i in set(queries['query_idx'])]
    os.makedirs(args.hits_dir, exist_ok=True)
    todo = [i for i in qidx if not os.path.exists(os.path.join(args.hits_dir, f'q{i:03d}.csv.gz'))]
    print(f'queries {qidx}; to do {todo}', flush=True)
    if not todo:
        return
    tag_str = ','.join(map(str, todo)) + f'|{args.seed}|{args.random_subsample}'
    if args.fp16_roundtrip:
        tag_str += '|fp16'
    tag = hashlib.sha1(tag_str.encode()).hexdigest()[:10]
    part_dir = os.path.join(args.partials_dir, tag)
    os.makedirs(part_dir, exist_ok=True)

    db = pd.read_csv(args.database, dtype=str)
    manifest = pd.read_csv(os.path.join(args.packed, 'manifest.csv'), dtype={'chain_id': str})
    ok = set(manifest.loc[manifest['status'] == 'ok', 'chain_id'])
    shard_of = dict(zip(manifest['chain_id'], manifest['shard'].astype(int)))

    # Pairs exactly as InferenceRunner._prepare_dataframe (database mode) builds them.
    runners, pairs = {}, []
    for qi in todo:
        q = queries[queries['query_idx'] == qi].iloc[0]
        r = InferenceRunner(checkpoint_path=args.checkpoint, base_save_dir=part_dir,
                            protein_database_search=(q.src_protein, q.src_chain, args.database),
                            tar_motif=q.src_motif, tar_ligand_id=q.src_ligand,
                            max_pLRMSD=-np.inf, max_pLRMSD_normed=-np.inf)
        r._prepare_dataframe()
        runners[qi] = r
        for j, ph in enumerate(r._pairs):
            pairs.append((qi, j, ph))
    runner = runners[todo[0]]
    runner._checkpoint_dir = os.path.dirname(args.checkpoint)
    if args.device:
        runner._device = args.device
    runner._load_model()

    # Query chains' features come from the per-file cache (identical content).
    query_store = {}
    for qi in todo:
        q = queries[queries['query_idx'] == qi].iloc[0]
        item, st = load_chain(args.cache, q.src_protein + q.src_chain)
        if item is None:
            raise RuntimeError(f'query {qi} has no features: {st}')
        query_store[q.src_protein + q.src_chain] = item
    if args.fp16_roundtrip:
        fp16_roundtrip(query_store)

    # One dataset over all pairs of the job (same construction as _prepare_dataloader).
    with open(os.path.join(runner._checkpoint_dir, 'dataset_config.yaml')) as f:
        dcfg = yaml.safe_load(f)
    pair_df = pd.DataFrame([ph.to_dict() for _, _, ph in pairs])
    pair_df['pair_idx'] = range(len(pairs))
    csv_path = os.path.join(part_dir, 'pairs.csv')
    pair_df.to_csv(csv_path, index=False)
    dcfg['args'].update(df_path=csv_path, base_data_path=os.path.join(args.cache, 'pdb_files'),
                        base_scannet_path=os.path.join(args.cache, 'scannet_embeddings'),
                        base_esm_embedding_path=os.path.join(args.cache, 'esm_embeddings'),
                        inference=True, ligand_column='ligand', tar_ligand_column='tar_ligand',
                        src_ligand_column='src_ligand')
    ds = PackedScanNetDataset(**dcfg['args'])
    assert len(ds) == len(pairs), 'dataset dropped pairs (invalid ligands?)'
    src_chain = (pair_df['src_protein'] + pair_df['src_chain']).values
    tar_chain = (pair_df['tar_protein'] + pair_df['tar_chain']).values
    valid = np.array([s in ok and t in query_store for s, t in zip(src_chain, tar_chain)])
    pair_shard = np.array([shard_of.get(s, -1) for s in src_chain])
    print(f'{len(pairs)} pairs, {int((~valid).sum())} without features; setup {time.time() - t0:.0f}s', flush=True)

    shards = sorted(set(pair_shard[valid]))
    if args.max_shards is not None:
        shards = shards[:args.max_shards]
    scores = {}
    todo_shards = [si for si in shards if not os.path.exists(os.path.join(part_dir, f'shard_{si:03d}.csv.gz'))]
    stager = ShardStager(args.packed, todo_shards, stage_root=None if args.no_stage else args.stage_root,
                         n_streams=args.stage_streams, log=lambda m: print(m, flush=True))
    pool, prefetch = ThreadPoolExecutor(max_workers=1), {}
    for si in shards:
        part_path = os.path.join(part_dir, f'shard_{si:03d}.csv.gz')
        if os.path.exists(part_path):
            p = pd.read_csv(part_path)
            scores.update({int(a): (b, c) for a, b, c in p[['pair_idx', 'pLRMSD', 'eLRMSD']].itertuples(index=False)})
            print(f'shard {si}: reused {len(p)} scores', flush=True)
            continue
        ts = time.time()
        store = prefetch.pop(si).result() if si in prefetch else load_shard(stager, si)
        nxt = [x for x in shards if x > si and not os.path.exists(os.path.join(part_dir, f'shard_{x:03d}.csv.gz'))]
        if nxt and nxt[0] not in prefetch:
            prefetch[nxt[0]] = pool.submit(load_shard, stager, nxt[0])  # overlaps with compute below
        if args.fp16_roundtrip:
            fp16_roundtrip(store)
        store.update(query_store)
        PackedScanNetDataset.STORE = store
        t_load = time.time() - ts
        idx = np.nonzero(valid & (pair_shard == si))[0].tolist()
        dl = DataLoader(Subset(ds, idx), batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=custom_collate_fn, pin_memory=True, shuffle=False)
        rows = []
        with torch.inference_mode():
            for batch in dl:
                out = runner._model.inference_step(batch)
                for b in range(len(out['metadata'])):
                    k = int(out['metadata'][b]['pair_idx'])
                    ph = pairs[k][2]
                    runner._process_visualization(ph, out, b)
                    rows.append((k, float(ph.pLRMSD), np.nan if ph.eLRMSD is None else float(ph.eLRMSD)))
        p = pd.DataFrame(rows, columns=['pair_idx', 'pLRMSD', 'eLRMSD'])
        assert len(p) == len(idx) and set(p['pair_idx']) == set(idx)
        atomic_csv(p, part_path)
        scores.update({a: (b, c) for a, b, c in rows})
        PackedScanNetDataset.STORE = {}
        del store
        print(f'shard {si}: {len(idx)} pairs, load {t_load:.0f}s, total {time.time() - ts:.0f}s '
              f'({len(idx) / (time.time() - ts):.1f} pairs/s)', flush=True)

    stager.finish()
    if args.max_shards is not None:
        # debug run: dump what we have for comparison, do not write final hit lists
        out = pair_df[['pair_idx', 'tar_protein', 'tar_chain', 'src_protein', 'src_chain', 'src_ligand']].copy()
        out['pLRMSD'] = out['pair_idx'].map(lambda k: scores.get(k, (np.nan, np.nan))[0])
        out['eLRMSD'] = out['pair_idx'].map(lambda k: scores.get(k, (np.nan, np.nan))[1])
        atomic_csv(out[out['pLRMSD'].notna()], os.path.join(part_dir, 'debug_scores.csv.gz'))
        print(f'debug scores written to {part_dir}/debug_scores.csv.gz')
        return

    # Final hit lists, built exactly as scripts/retrieval_search.py (DB row order, stable sort).
    clusters = pd.read_csv(args.clusters, dtype=str)
    cmap = dict(zip(clusters['chain_id'], clusters['cluster_id']))
    for qi in todo:
        q = queries[queries['query_idx'] == qi].iloc[0]
        ks = [k for k, (a, _, _) in enumerate(pairs) if a == qi]
        res = pair_df.iloc[ks].copy()
        res['pLRMSD'] = [scores.get(k, (np.nan, np.nan))[0] for k in ks]
        res['eLRMSD'] = [scores.get(k, (np.nan, np.nan))[1] for k in ks]
        res['chain_id'] = res['src_protein'] + res['src_chain']
        res = res[res['chain_id'] != q.src_protein + q.src_chain]
        hits = pd.DataFrame({
            'chain_id': res['chain_id'],
            'cluster_id': res['chain_id'].map(cmap),
            'ligand': res['src_ligand'],
            'normalized_pLRMSD': res['pLRMSD'].astype(float) / res['eLRMSD'].astype(float),
        }).sort_values('normalized_pLRMSD', ascending=True, na_position='last', kind='mergesort')
        atomic_csv(hits, os.path.join(args.hits_dir, f'q{qi:03d}.csv.gz'))
        print(f'query {qi}: wrote {len(hits)} hits ({int(hits.normalized_pLRMSD.isna().sum())} without score)')
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
