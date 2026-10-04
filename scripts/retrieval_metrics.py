"""Retrieval benchmark: metrics from the per-query ranked hit lists.

Hits are ranked by --score (normalized_pLRMSD = pLRMSD / eLRMSD by default, or
pLRMSD) ascending, with the query chain and (unless --keep_same_pdb) the other
chains of the query's PDB entry removed. Hits without a score (pairs that failed
feature loading) are ignored. A hit "has the same ligand" if its src_ligand equals
the query's src_ligand. It is a "homolog" of the query if its Foldseek (TM >= 0.6)
cluster_id equals the query's cluster_id (chains with no cluster never match) or,
with --homologs, if its TM-score to the query, max(qtmscore, ttmscore) from a
Foldseek query-vs-database search, is >= --homolog_tm. Cluster membership alone
misses homologs assigned to another cluster representative.

  Top_K_retrieval               fraction of queries with >= 1 of the top-K hits
                                having the same ligand OR being a homolog.
  Top_K_retrieval_not_FoldSeek  drop all homologs of the query; fraction of
                                queries with >= 1 of the top-K remaining hits
                                having the same ligand.
metrics.csv has one block per stratum: 'all' (headline numbers) and queries whose
ligand has < 10 / >= 10 heavy atoms (src_ligand_n_atoms of the query entry).
Ties in the score (common: the calibration models are boosted trees) are broken
by database order in the two main columns; the *_random_ties columns give the
expectation under uniformly random tie-breaking.

Inputs
  --hits_dir      <hits_dir>/qIII.csv.gz from scripts/retrieval_search.py or
                  scripts/retrieval_fast.py, sorted by normalized_pLRMSD in
                  database order. With another --score the file needs that
                  column; it is re-sorted stably, so ties in that score are then
                  broken by normalized_pLRMSD order instead of database order.
  --partials_dir  rebuild each query's full hit list from the per-shard scores
                  of scripts/retrieval_fast.py (<partials_dir>/<tag>/pairs.csv and
                  shard_XXX.csv.gz), in database order, and rank it by --score.
                  Pairs are mapped to queries by (tar_protein, tar_chain,
                  tar_ligand, tar_motif); tag dirs of the same query are merged.
                  A query is used only if every pair whose DB chain is 'ok' in
                  --manifest has a score, and, when --hits_dir is also given,
                  only if <hits_dir>/qIII.csv.gz exists. --check_against_hits
                  asserts that the rebuilt normalized_pLRMSD lists match the
                  hit files (see check_against_hits: partial scores have 6
                  decimals, so near-ties can be ordered differently).
Outputs go to --out_dir (default <work>/results): metrics.csv, per_query.csv and,
with --export, hits_top200.csv.gz. The compact list keeps, per query, the top-200
hits overall plus the top-200 hits outside the query's cluster, so every metric
here is exactly reproducible from it.

Usage
  python3 scripts/retrieval_metrics.py --hits_dir <work>/hits_fast --export
  python3 scripts/retrieval_metrics.py --partials_dir <work>/fast_partials \\
      --hits_dir <work>/hits_fast --check_against_hits
  python3 scripts/retrieval_metrics.py --partials_dir <work>/fast_partials --score pLRMSD
"""
import argparse
import ast
import glob
import os
import re
import sys
from math import comb

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'aligner_dl'))
from utils.constants import (  # noqa: E402
    RETRIEVAL_CLUSTERS_CSV,
    RETRIEVAL_HIT_FILE_FORMAT,
    RETRIEVAL_HIT_FILE_PATTERN,
    RETRIEVAL_HOMOLOG_MIN_TM,
    RETRIEVAL_LIGAND_SIZE_CUT,
    RETRIEVAL_PACKED_MANIFEST,
    RETRIEVAL_PARTIAL_PAIRS_CSV,
    RETRIEVAL_PARTIAL_SHARD_PATTERN,
    RETRIEVAL_QUERIES_CSV,
    RETRIEVAL_RESULTS_DIR,
    RETRIEVAL_TOP_KS,
)

KEEP = 200
SCORES = ('normalized_pLRMSD', 'pLRMSD')
SCORE_COLUMNS = ('normalized_pLRMSD', 'pLRMSD', 'eLRMSD')
COMPACT_NAME = 'hits_top200.csv.gz'
METRICS_NAME = 'metrics.csv'
PER_QUERY_NAME = 'per_query.csv'
QUERY_KEY = ['src_protein', 'src_chain', 'src_ligand', 'src_motif']
TARGET_KEY = ['tar_protein', 'tar_chain', 'tar_ligand', 'tar_motif']
HIT_DTYPES = {'chain_id': str, 'cluster_id': str, 'ligand': str}
QUERY_DTYPES = {c: str for c in ('chain_id', 'src_protein', 'src_chain', 'src_ligand', 'src_motif', 'cluster_id')}


def load_hit_files(
    hits_dir: str,
    score: str,
) -> dict[int, pd.DataFrame]:
    """Read <hits_dir>/qIII.csv.gz; return {query_idx: hits sorted by `score`}."""
    out = {}
    for path in sorted(glob.glob(os.path.join(hits_dir, RETRIEVAL_HIT_FILE_PATTERN))):
        h = pd.read_csv(path, dtype=HIT_DTYPES)
        if score not in h.columns:
            raise ValueError(f'{path} has no {score} column; use --partials_dir')
        if score != 'normalized_pLRMSD':
            h = h.sort_values(score, ascending=True, na_position='last', kind='mergesort')
        out[int(os.path.basename(path)[1:4])] = h.reset_index(drop=True)
    return out


def rebuild_from_partials(
    partials_dir: str,
    queries: pd.DataFrame,
    clusters_csv: str,
    manifest_csv: str | None,
    score: str,
    only: set[int] | None = None,
) -> dict[int, pd.DataFrame]:
    """Rebuild full per-query hit lists from scripts/retrieval_fast.py partial scores.

    Every <partials_dir>/<tag>/ with a readable pairs.csv contributes its pairs; the
    scores come from the shard_XXX.csv.gz files present (pair_idx, pLRMSD, eLRMSD).
    Each query is taken from a single tag dir, never merged across dirs (another dir
    can hold an earlier run of the same query with different settings): the dir in
    which every pair whose DB chain has status 'ok' in `manifest_csv` is scored; if
    several qualify, the most recently modified one; without a manifest, the dir with
    the most scored pairs. Queries are kept if they are in `only` (when given) and a
    complete dir exists. A query whose pairs were written with an empty tar_motif (a
    motif the inference parser could not read) is matched on protein, chain and ligand
    when that is unique. Returns {query_idx: frame} with chain_id, cluster_id,
    ligand, normalized_pLRMSD, pLRMSD and eLRMSD, in database order with the query
    chain removed, stably sorted by `score` with unscored pairs last.
    """
    qmap = {
        (r.src_protein, r.src_chain, r.src_ligand, parse_motif(r.src_motif)): int(r.query_idx)
        for r in queries.itertuples()
    }
    site_counts = queries.groupby(['src_protein', 'src_chain', 'src_ligand']).size()
    qmap_no_motif = {
        (r.src_protein, r.src_chain, r.src_ligand): int(r.query_idx)
        for r in queries.itertuples()
        if site_counts[(r.src_protein, r.src_chain, r.src_ligand)] == 1
    }
    matched_without_motif = set()

    def query_of(
        protein: str,
        chain: str,
        ligand: str,
        motif: str,
    ) -> int:
        motif = parse_motif(motif) if isinstance(motif, str) else ()
        qi = qmap.get((protein, chain, ligand, motif), -1)
        if qi < 0 and not motif:
            qi = qmap_no_motif.get((protein, chain, ligand), -1)
            if qi >= 0:
                matched_without_motif.add(qi)
        return qi

    frames = []
    for tag_dir in sorted(glob.glob(os.path.join(partials_dir, '*', ''))):
        try:
            pairs = pd.read_csv(os.path.join(tag_dir, RETRIEVAL_PARTIAL_PAIRS_CSV), dtype=str,
                                usecols=TARGET_KEY + QUERY_KEY + ['pair_idx'])
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as e:
            print(f'skipping {tag_dir}: {e!r}')
            continue
        pairs['pair_idx'] = pairs['pair_idx'].astype(int)
        pairs['query_idx'] = [query_of(a, b, c, d) for a, b, c, d in pairs[TARGET_KEY].itertuples(index=False)]
        pairs = pairs[pairs['query_idx'] >= 0]
        if only is not None:
            pairs = pairs[pairs['query_idx'].isin(only)]
        if pairs.empty:
            continue
        pairs['db_order'] = pairs.groupby('query_idx').cumcount()
        pairs['tag_dir'] = tag_dir
        pairs['tag_mtime'] = os.path.getmtime(tag_dir)
        shard_paths = sorted(glob.glob(os.path.join(tag_dir, RETRIEVAL_PARTIAL_SHARD_PATTERN)))
        if shard_paths:
            scores = pd.concat([pd.read_csv(p) for p in shard_paths], ignore_index=True)
        else:
            scores = pd.DataFrame(columns=['pair_idx', 'pLRMSD', 'eLRMSD'])
        scores['pair_idx'] = scores['pair_idx'].astype(int)
        frames.append(pairs.merge(scores[['pair_idx', 'pLRMSD', 'eLRMSD']], on='pair_idx', how='left'))
    if matched_without_motif:
        print(f'queries {sorted(matched_without_motif)} were scored without a motif (empty tar_motif in '
              f'pairs.csv) and are matched on protein, chain and ligand')
    if not frames:
        return {}
    allp = pd.concat(frames, ignore_index=True)
    allp['scored'] = allp['pLRMSD'].notna()
    allp['chain_id'] = allp['src_protein'] + allp['src_chain']
    ok_chains = None
    if manifest_csv and os.path.exists(manifest_csv):
        man = pd.read_csv(manifest_csv, dtype={'chain_id': str})
        ok_chains = set(man.loc[man['status'] == 'ok', 'chain_id'])
    else:
        print(f'WARNING: no manifest {manifest_csv}; completeness of partial scores is not checked')
    cmap = pd.read_csv(clusters_csv, dtype=str).set_index('chain_id')['cluster_id']
    q = queries.set_index('query_idx')
    out, incomplete = {}, []
    for qi, pq in allp.groupby('query_idx', sort=True):
        candidates = []
        for tag_dir, p in pq.groupby('tag_dir', sort=False):
            complete = ok_chains is None or p.loc[p['chain_id'].isin(ok_chains), 'scored'].all()
            if complete:
                candidates.append((int(p['scored'].sum()), p['tag_mtime'].iat[0], tag_dir, p))
        if not candidates:
            incomplete.append(int(qi))
            continue
        key = (lambda c: c[1]) if ok_chains is not None else (lambda c: c[0])
        p = max(candidates, key=key)[3].sort_values('db_order')
        p = p[p['chain_id'] != q.at[qi, 'src_protein'] + q.at[qi, 'src_chain']]
        plrmsd, elrmsd = p['pLRMSD'].astype(float), p['eLRMSD'].astype(float)
        h = pd.DataFrame({
            'chain_id': p['chain_id'].values,
            'cluster_id': p['chain_id'].map(cmap).values,
            'ligand': p['src_ligand'].values,
            'normalized_pLRMSD': (plrmsd / elrmsd).values,
            'pLRMSD': plrmsd.values,
            'eLRMSD': elrmsd.values,
        })
        out[int(qi)] = h.sort_values(score, ascending=True, na_position='last', kind='mergesort') \
            .reset_index(drop=True)
    if incomplete:
        print(f'partial scores incomplete for queries {incomplete}; they are excluded')
    return out


def check_against_hits(
    rebuilt: dict[int, pd.DataFrame],
    hits_dir: str,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """Assert that each rebuilt list matches <hits_dir>/qIII.csv.gz.

    The partial scores are stored with 6 decimals while the hit files were sorted on
    full-precision scores, so hits whose normalized_pLRMSD differ by ~1e-6 can swap.
    A query passes if its chain_id order is identical, or if every (chain_id, ligand)
    has the same normalized_pLRMSD in both lists within `rtol`/`atol` (both lists are
    sorted by it, so the orders then differ only within such near-ties).
    """
    exact, near, bad = [], [], []
    for qi, h in rebuilt.items():
        ref = pd.read_csv(os.path.join(hits_dir, RETRIEVAL_HIT_FILE_FORMAT.format(qi)), dtype=HIT_DTYPES)
        if ref['chain_id'].tolist() == h['chain_id'].tolist():
            exact.append(qi)
        elif same_hit_scores(h, ref, rtol, atol):
            near.append(qi)
        else:
            bad.append(qi)
    print(f'check_against_hits: {len(exact) + len(near)}/{len(rebuilt)} queries match the hit files '
          f'({len(exact)} identical order, {len(near)} {near} differing only within near-ties)')
    assert not bad, f'rebuilt hits differ from the hit files for queries {bad}'


def rank_hits(
    per_query: dict[int, pd.DataFrame],
    score: str,
) -> pd.DataFrame:
    """Drop unscored hits and add query_idx and rank (1-based, in list order)."""
    frames = []
    for qi, h in sorted(per_query.items()):
        h = h[h[score].notna()].reset_index(drop=True)
        h.insert(0, 'rank', range(1, len(h) + 1))
        h.insert(0, 'query_idx', qi)
        frames.append(h)
    return pd.concat(frames, ignore_index=True)


def annotate(
    hits: pd.DataFrame,
    queries: pd.DataFrame,
    homolog_pairs: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Mark hits with the query's ligand (same_ligand) and structural homologs of the query (homolog).

    A hit is a homolog if it is in the query's Foldseek cluster, or if (query chain, hit
    chain) is in `homolog_pairs` (TM-score to the query at or above the threshold).
    """
    q = queries.set_index('query_idx')
    hits = hits.copy()
    hits['same_ligand'] = hits['ligand'].values == q.loc[hits['query_idx'], 'src_ligand'].values
    hits['same_cluster'] = (hits['cluster_id'].values == q.loc[hits['query_idx'], 'cluster_id'].values) \
        & hits['cluster_id'].notna().values
    hits['homolog'] = hits['same_cluster']
    if homolog_pairs is not None:
        query_chain = q.loc[hits['query_idx'], 'chain_id'].values
        hits['homolog'] = hits['homolog'].values | np.array(
            [(a, b) in homolog_pairs for a, b in zip(query_chain, hits['chain_id'].values)], dtype=bool)
    return hits


def load_homolog_pairs(
    m8_path: str,
    min_tm: float,
) -> set[tuple[str, str]]:
    """(query chain, DB chain) pairs with max(qtmscore, ttmscore) >= min_tm in a Foldseek
    easy-search table (columns query,target,fident,alntmscore,qtmscore,ttmscore,...)."""
    cols = ['query', 'target', 'fident', 'alntmscore', 'qtmscore', 'ttmscore']
    m8 = pd.read_csv(m8_path, sep='\t', header=None, usecols=range(len(cols)), names=cols)
    strip = lambda s: s.str.replace(r'\.pdb$', '', regex=True)  # noqa: E731
    keep = m8[np.maximum(m8['qtmscore'], m8['ttmscore']) >= min_tm]
    return set(zip(strip(keep['query']), strip(keep['target'])))


def compact(
    hits: pd.DataFrame,
) -> pd.DataFrame:
    keep = []
    for _, h in hits.groupby('query_idx', sort=True):
        h = h.sort_values('rank')
        keep.append(pd.concat([h.head(KEEP), h[~h['homolog']].head(KEEP)]).drop_duplicates('rank'))
    return pd.concat(keep).sort_values(['query_idx', 'rank'])


def p_hit_random_ties(
    scores,
    pos,
    k: int,
) -> float:
    """P(>=1 positive in the top-k) when ties in score are broken uniformly at random.

    `scores` must be sorted ascending. The calibrated pLRMSD / eLRMSD models are
    gradient-boosted trees, so scores are piecewise constant and ties are common. The
    main metric breaks ties by the stable database order; this is the
    tie-order-independent expectation.
    """
    scores, pos = np.asarray(scores), np.asarray(pos, bool)
    if len(scores) <= k:
        return float(pos.any())
    s_k = scores[k - 1]
    better = scores < s_k
    if pos[better].any():
        return 1.0
    tied = scores == s_k
    if tied[-1]:
        raise ValueError('tie group reaches the end of the available hit list')
    g, p, m = int(tied.sum()), int(pos[tied].sum()), k - int(better.sum())
    return 1.0 - comb(g - p, m) / comb(g, m)


def metrics(
    hits: pd.DataFrame,
    queries: pd.DataFrame,
    score: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-stratum metrics and per-query rows; query ligand size is the sum of src_ligand_n_atoms."""
    per_q = []
    for qi in queries['query_idx']:
        h = hits[hits['query_idx'] == qi].sort_values('rank')
        row = {'query_idx': qi, 'n_hits_scored': len(h)}
        nc = h[~h['homolog']]
        for k in RETRIEVAL_TOP_KS:
            top, top_nc = h.head(k), nc.head(k)
            row[f'top{k}_hit'] = bool((top['same_ligand'] | top['homolog']).any())
            row[f'top{k}_hit_not_foldseek'] = bool(top_nc['same_ligand'].any())
            row[f'top{k}_p_random_ties'] = p_hit_random_ties(
                h[score], h['same_ligand'] | h['homolog'], k)
            row[f'top{k}_p_random_ties_not_foldseek'] = p_hit_random_ties(
                nc[score], nc['same_ligand'], k)
        lig = h[h['same_ligand']]
        row['first_same_ligand_rank'] = int(lig['rank'].iloc[0]) if len(lig) else None
        lig_nc = nc[nc['same_ligand']]
        row['first_same_ligand_rank_not_foldseek'] = int(lig_nc['rank'].iloc[0]) if len(lig_nc) else None
        per_q.append(row)
    per_q = pd.DataFrame(per_q)
    n_atoms = queries.set_index('query_idx')['src_ligand_n_atoms'].map(lambda v: sum(ast.literal_eval(str(v))))
    per_q['query_ligand_n_atoms'] = per_q['query_idx'].map(n_atoms)
    cut = RETRIEVAL_LIGAND_SIZE_CUT
    strata = [('all', per_q),
              (f'ligand<{cut}', per_q[per_q['query_ligand_n_atoms'] < cut]),
              (f'ligand>={cut}', per_q[per_q['query_ligand_n_atoms'] >= cut])]
    rows = []
    for name, pq in strata:
        for k in RETRIEVAL_TOP_KS:
            rows.append({'stratum': name, 'K': k,
                         'Top_K_retrieval': pq[f'top{k}_hit'].mean(),
                         'Top_K_retrieval_not_FoldSeek': pq[f'top{k}_hit_not_foldseek'].mean(),
                         'Top_K_retrieval_random_ties': pq[f'top{k}_p_random_ties'].mean(),
                         'Top_K_retrieval_not_FoldSeek_random_ties': pq[f'top{k}_p_random_ties_not_foldseek'].mean(),
                         'n_queries': len(pq)})
    return pd.DataFrame(rows), per_q


def same_hit_scores(
    a: pd.DataFrame,
    b: pd.DataFrame,
    rtol: float,
    atol: float,
) -> bool:
    """True if `a` and `b` hold the same (chain_id, ligand) hits with close normalized_pLRMSD."""
    if len(a) != len(b):
        return False
    key = ['chain_id', 'ligand', 'normalized_pLRMSD']
    sa = a.sort_values(key, kind='mergesort', na_position='last')
    sb = b.sort_values(key, kind='mergesort', na_position='last')
    if not (sa[key[:2]].values == sb[key[:2]].values).all():
        return False
    return bool(np.allclose(sa[key[2]].values, sb[key[2]].values, rtol=rtol, atol=atol, equal_nan=True))


def parse_motif(
    val,
) -> tuple[int | str, ...]:
    """Residue numbers of a motif string such as '[1, 2, 3]' or '1,2,3', as ints.

    A residue with an insertion code (e.g. '95C') maps to its residue number (95), as in
    InferenceRunner._parse_motif; tokens that are not residues are kept as strings.
    """
    tokens = (x.strip().strip('\'"') for x in str(val).strip('[]() ').split(','))
    matches = ((x, re.fullmatch(r'(-?\d+)[A-Za-z]?', x)) for x in tokens if x)
    return tuple(int(m.group(1)) if m else x for x, m in matches)


def patch_hits(
    per_query: dict[int, pd.DataFrame],
    patch_dirs: list[str],
    score: str,
) -> dict[int, pd.DataFrame]:
    """Replace scores with those of re-scored entries and re-rank.

    For each query with a hit list in a patch dir, the rows of the same (chain_id, ligand)
    take the patch's normalized_pLRMSD, pLRMSD and eLRMSD (whichever the patch has). A patch
    that covers every row of the query replaces its list. Lists are then stably re-sorted by
    `score`, so ties keep their previous order (database order for unpatched rows). A patch
    row missing from the query's list raises.
    """
    key = ['chain_id', 'ligand']
    patched = {}
    for patch_dir in patch_dirs:
        for path in sorted(glob.glob(os.path.join(patch_dir, RETRIEVAL_HIT_FILE_PATTERN))):
            qi = int(os.path.basename(path)[1:4])
            if qi not in per_query:
                continue
            patch = pd.read_csv(path, dtype={'chain_id': str, 'cluster_id': str, 'ligand': str})
            base = per_query[qi].set_index(key)
            pidx = patch.set_index(key)
            unknown = pidx.index.difference(base.index)
            if len(unknown):
                raise ValueError(f'{path}: {len(unknown)} entries not in the hit list of query {qi}')
            cols = [c for c in SCORE_COLUMNS if c in pidx.columns]
            if base.index.isin(pidx.index).all():
                base = pidx[['cluster_id'] + cols].copy()
                patched[qi] = 'replaced'
            else:
                for c in cols:
                    if c not in base.columns:
                        base[c] = np.nan
                    base.loc[pidx.index, c] = pidx[c].values
                patched[qi] = len(pidx)
            per_query[qi] = base.reset_index() \
                .sort_values(score, na_position='last', kind='mergesort').reset_index(drop=True)
    replaced = sorted(q for q, v in patched.items() if v == 'replaced')
    print(f'patched {len(patched)} queries' + (f'; replaced the lists of queries {replaced}' if replaced else ''))
    return per_query


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hits_dir', default=None, help='per-query hit lists; required unless --partials_dir is given')
    ap.add_argument('--partials_dir', default=None, help='rebuild the hit lists from retrieval_fast.py partial scores')
    ap.add_argument('--score', choices=SCORES, default='normalized_pLRMSD', help='column the hits are ranked by')
    ap.add_argument('--check_against_hits', action='store_true',
                    help='assert the rebuilt order equals the hit files (needs --partials_dir, --hits_dir and '
                         '--score normalized_pLRMSD)')
    ap.add_argument('--out_dir', default=RETRIEVAL_RESULTS_DIR)
    ap.add_argument('--export', action='store_true', help=f'write <out_dir>/{COMPACT_NAME}')
    ap.add_argument('--queries', default=RETRIEVAL_QUERIES_CSV)
    ap.add_argument('--clusters', default=RETRIEVAL_CLUSTERS_CSV)
    ap.add_argument('--patch_hits_dir', nargs='+', default=None,
                    help='hit lists of re-scored entries (retrieval_fast.py on a database subset, or a full '
                         're-run of a query); their scores replace those of the same (chain_id, ligand)')
    ap.add_argument('--homologs', default=None,
                    help='Foldseek easy-search table of query vs DB chains; hits with TM-score >= --homolog_tm '
                         'to the query count as homologs in addition to the query cluster')
    ap.add_argument('--homolog_tm', type=float, default=RETRIEVAL_HOMOLOG_MIN_TM)
    ap.add_argument('--keep_same_pdb', action='store_true',
                    help="keep the other chains of the query's PDB entry (dropped by default)")
    ap.add_argument('--manifest', default=RETRIEVAL_PACKED_MANIFEST,
                    help='packing manifest, used to check that partial scores are complete')
    args = ap.parse_args()
    if not args.hits_dir and not args.partials_dir:
        ap.error('--hits_dir is required unless --partials_dir is given')
    if args.check_against_hits and not (args.hits_dir and args.partials_dir and args.score == 'normalized_pLRMSD'):
        ap.error('--check_against_hits needs --partials_dir, --hits_dir and --score normalized_pLRMSD')

    queries = pd.read_csv(args.queries, dtype=QUERY_DTYPES)
    if args.partials_dir:
        only = None
        if args.hits_dir:
            only = {int(os.path.basename(p)[1:4])
                    for p in glob.glob(os.path.join(args.hits_dir, RETRIEVAL_HIT_FILE_PATTERN))}
        per_query = rebuild_from_partials(args.partials_dir, queries, args.clusters, args.manifest,
                                          args.score, only=only)
        if args.check_against_hits:
            check_against_hits(per_query, args.hits_dir)
    else:
        per_query = load_hit_files(args.hits_dir, args.score)
    if not per_query:
        raise SystemExit('no hit lists found')
    if args.patch_hits_dir:
        per_query = patch_hits(per_query, args.patch_hits_dir, args.score)
    if not args.keep_same_pdb:
        pdb_of = queries.set_index('query_idx')['src_protein']
        per_query = {qi: h[h['chain_id'].str[:4] != pdb_of[qi]].reset_index(drop=True) for qi, h in per_query.items()}
    homolog_pairs = load_homolog_pairs(args.homologs, args.homolog_tm) if args.homologs else None
    hits = annotate(rank_hits(per_query, args.score), queries, homolog_pairs)
    missing = sorted(set(queries['query_idx']) - set(hits['query_idx']))
    if missing:
        print(f'WARNING: no hit list for queries {missing}; they are excluded')
        queries = queries[~queries['query_idx'].isin(missing)]

    os.makedirs(args.out_dir, exist_ok=True)
    if args.export:
        c = compact(hits)
        cols = ['query_idx', 'rank', 'chain_id', 'cluster_id', 'ligand'] + [s for s in SCORE_COLUMNS if s in c.columns]
        out_path = os.path.join(args.out_dir, COMPACT_NAME)
        c[cols].to_csv(out_path, index=False, float_format='%.5f')
        print(f'Wrote {len(c)} rows to {out_path}')

    m, per_q = metrics(hits, queries, args.score)
    m.to_csv(os.path.join(args.out_dir, METRICS_NAME), index=False, float_format='%.4f')
    per_q.merge(queries[['query_idx', 'chain_id', 'src_ligand', 'cluster_id', 'cluster_size']],
                on='query_idx').to_csv(os.path.join(args.out_dir, PER_QUERY_NAME), index=False)
    print(f'ranked by {args.score}')
    print(m.to_string(index=False))


if __name__ == '__main__':
    main()
