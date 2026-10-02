"""Sample random negative chain pairs (chains that do NOT bind a common ligand)
from the test-set chains of one split, and write an inference.py --csv_path input.

Chain pool: every chain in the test CSV that appears in a pair with a valid ligand
(ALL_INVALID_LIGANDS removed, exactly as the ScanNetDataset does for the positives).

The ligand set of a chain, used for the no-overlap check, is the union of:
  1. every ligand the chain binds in the test CSV (as tar or src, any row);
  2. every ligand listed for that PDB chain in BioLiP_nr (--biolip_nr, optional);
  3. `ligand_all` of every row of BIOLIP_NR_DB_PATH whose `id_all` contains that
     chain (ligands of the whole 100%-identity sequence cluster; conservative).
Crystallization/invalid ligands are ignored for the overlap check, since they are
also excluded from the positives.

A pair is kept only if the ligand sets are disjoint, the two chains come from
different PDB entries, the unordered pair is not a positive pair and it has not
been sampled already. Sampling is uniform over ordered (tar, src) pairs.

Each chain is evaluated with one ligand it binds in the test set, chosen with the
same seeded RNG among its valid test-set ligands; it decides the prepared PDB files.
As for the positives, no motif is given.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'aligner_dl'))
from utils.constants import ALL_INVALID_LIGANDS, BIOLIP_NR_DB_PATH  # noqa: E402

INVALID = set(ALL_INVALID_LIGANDS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True)
    parser.add_argument('--test_csv', required=True, help='test-set pairs CSV')
    parser.add_argument('--positives_csv', required=True, help='evaluated positives (baseline.csv)')
    parser.add_argument('--biolip_nr', default=None, help='BioLiP_nr txt (tab separated, no header)')
    parser.add_argument('--biolip_nr_db', default=BIOLIP_NR_DB_PATH)
    parser.add_argument('--n', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_csv', required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    test = pd.read_csv(args.test_csv)
    positives = pd.read_csv(args.positives_csv)
    lig_sets, valid_ligs, n_atoms = collect_test_ligands(test)
    pool = sorted(valid_ligs)
    n_from_test = {ch: len(s) for ch, s in lig_sets.items()}
    add_biolip_ligands(lig_sets, args.biolip_nr, args.biolip_nr_db)
    overlap_sets = {ch: s - INVALID - {'nan', ''} for ch, s in lig_sets.items()}

    pos_pairs = set()
    for _, r in pd.concat([test, positives]).iterrows():
        a = (r['tar_protein'].lower(), r['tar_chain'])
        b = (r['src_protein'].lower(), r['src_chain'])
        pos_pairs.add(frozenset([a, b]))

    chain_lig = {ch: sorted(valid_ligs[ch])[rng.integers(len(valid_ligs[ch]))] for ch in pool}
    rows, rejections, n_draws = sample_pairs(pool, overlap_sets, pos_pairs, chain_lig, n_atoms, args.n, rng)
    out = pd.DataFrame(rows)
    assert not any(
        frozenset([(r.tar_protein, r.tar_chain), (r.src_protein, r.src_chain)]) in pos_pairs
        for r in out.itertuples()
    )
    assert all(
        not (set(r.tar_ligands_all.split('|')) & set(r.src_ligands_all.split('|')) - {''})
        for r in out.itertuples()
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(
        f'[{args.split}] pool={len(pool)} chains; draws={n_draws}; rejections={rejections}; '
        f'kept={len(out)}; mean ligands/chain test-only={np.mean(list(n_from_test.values())):.2f} '
        f'with BioLiP={np.mean([len(overlap_sets[c]) for c in pool]):.2f} -> {args.out_csv}'
    )


def first_n_atoms(x) -> int:
    return int(str(x).strip()[1:-1].split(',')[0])


def collect_test_ligands(test: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Per-chain ligand sets from the test CSV.

    Returns (all ligands per chain, valid ligands per chain, atom count per (chain, ligand)).
    """
    lig_sets, valid_ligs, n_atoms = {}, {}, {}
    for _, r in test.iterrows():
        for role in ['tar', 'src']:
            ch = (r[f'{role}_protein'].lower(), r[f'{role}_chain'])
            lig = r['ligand_id']
            lig_sets.setdefault(ch, set()).add(lig)
            if lig not in INVALID:
                valid_ligs.setdefault(ch, set()).add(lig)
                n_atoms.setdefault((ch, lig), first_n_atoms(r[f'{role}_ligand_n_atoms']))
    return lig_sets, valid_ligs, n_atoms


def add_biolip_ligands(
    lig_sets: dict,
    biolip_nr: str | None,
    biolip_nr_db: str | None,
) -> None:
    """Extend `lig_sets` in place with the BioLiP_nr per-chain ligands and the sequence-cluster ligand_all."""
    if biolip_nr and os.path.exists(biolip_nr):
        bl = pd.read_csv(biolip_nr, sep='\t', header=None, usecols=[0, 1, 4], dtype=str)
        for p, c, lig in bl.itertuples(index=False):
            ch = (str(p).lower(), str(c))
            if ch in lig_sets:
                lig_sets[ch].add(str(lig))
    if biolip_nr_db and os.path.exists(biolip_nr_db):
        db = pd.read_csv(biolip_nr_db, usecols=['ligand_all', 'id_all'], dtype=str).dropna()
        for ligs, ids in db.itertuples(index=False):
            lset = set(ligs.split('|'))
            for i in ids.split('|'):
                p, _, c = i.partition('_')
                ch = (p.lower(), c)
                if ch in lig_sets:
                    lig_sets[ch] |= lset


def sample_pairs(
    pool: list,
    overlap_sets: dict,
    pos_pairs: set,
    chain_lig: dict,
    n_atoms: dict,
    n: int,
    rng: np.random.Generator,
) -> tuple[list[dict], dict, int]:
    """Rejection-sample `n` ordered chain pairs; returns (rows, rejection counts, number of draws)."""
    rows, seen, n_draws = [], set(), 0
    rejections = {'same_pdb': 0, 'shared_ligand': 0, 'positive': 0, 'dup': 0}
    while len(rows) < n:
        i, j = rng.integers(len(pool), size=2)
        n_draws += 1
        a, b = pool[i], pool[j]
        key = frozenset([a, b])
        if a[0] == b[0]:
            rejections['same_pdb'] += 1
            continue
        if overlap_sets[a] & overlap_sets[b]:
            rejections['shared_ligand'] += 1
            continue
        if key in pos_pairs:
            rejections['positive'] += 1
            continue
        if key in seen:
            rejections['dup'] += 1
            continue
        seen.add(key)
        la, lb = chain_lig[a], chain_lig[b]
        rows.append({
            'tar_protein': a[0], 'tar_chain': a[1], 'src_protein': b[0], 'src_chain': b[1],
            'tar_ligand': la, 'src_ligand': lb,
            'tar_ligand_n_atoms': f'[{n_atoms[(a, la)]}]', 'src_ligand_n_atoms': f'[{n_atoms[(b, lb)]}]',
            'tar_ligands_all': '|'.join(sorted(overlap_sets[a])),
            'src_ligands_all': '|'.join(sorted(overlap_sets[b])),
        })
    return rows, rejections, n_draws


if __name__ == '__main__':
    main()
