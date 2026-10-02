"""Run scripts/inference.py (--csv_path mode) on a pair list, reusing the exact
input features that were used to evaluate the positives.

Before calling inference.py the inference cache (<base_save_dir>/.cache) is filled
with symlinks to the precomputed inputs of the validation run that produced the
positives: ScanNet features, prepared PDBs and ESM embeddings. Missing items are
computed by inference.py as usual. The Chimera visualisation step of inference.py
is replaced by a no-op; it does not affect the model outputs.

inference.py writes <base_save_dir>/<ckpt_dir_name>_<timestamp>/inference_results.csv,
whose raw outputs _embedding, _gap, _corr_rmsd, _radius correspond to the positives'
embedding_similarity, entropy, corr_rmsd and radius_of_gyration.
"""
import argparse
import hashlib
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'aligner_dl'))
from utils.constants import ESM_CACHE_NAME, LIGAND_DIR, SCANNET_DIR  # noqa: E402

PAIR_COLUMNS = [
    'tar_protein', 'tar_chain', 'src_protein', 'src_chain', 'tar_ligand', 'src_ligand',
    'tar_ligand_n_atoms', 'src_ligand_n_atoms',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs_csv', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--base_save_dir', required=True)
    parser.add_argument('--scannet_dir', default=SCANNET_DIR)
    parser.add_argument('--ligand_dir', default=LIGAND_DIR)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.pairs_csv, dtype=str)
    if args.limit:
        df = df.iloc[:args.limit]
    os.makedirs(args.base_save_dir, exist_ok=True)
    pairs_path = os.path.join(args.base_save_dir, 'pairs_input.csv')
    df[[c for c in PAIR_COLUMNS if c in df.columns]].to_csv(pairs_path, index=False)
    fill_cache(df, os.path.join(args.base_save_dir, '.cache'), args.scannet_dir, args.ligand_dir)

    sys.path.insert(0, os.getcwd())
    import scripts.inference as inf
    inf.process_alignment = lambda **kwargs: None
    sys.argv = [
        'inference.py',
        '--csv_path', pairs_path,
        '--checkpoint', args.checkpoint,
        '--base_save_dir', args.base_save_dir,
    ]
    inf.main()


def link(
    src: str,
    dst: str,
) -> int:
    """Symlink `src` to `dst` if `src` exists and `dst` does not; returns 1 if a link was made."""
    if os.path.exists(src) and not os.path.lexists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.symlink(src, dst)
        return 1
    return 0


def fill_cache(
    df: pd.DataFrame,
    cache: str,
    scannet_dir: str,
    ligand_dir: str,
) -> None:
    """Link the ScanNet features, prepared PDBs and ESM embeddings of every chain in `df` into `cache`."""
    counts = {'scannet': 0, 'pdb': 0, 'esm': 0}
    chains = set()
    for r in df.itertuples():
        chains.add((r.tar_protein, r.tar_chain, r.tar_ligand))
        chains.add((r.src_protein, r.src_chain, r.src_ligand))
    for pdb, chain, ligand in chains:
        name = f'{pdb}{chain}'
        counts['scannet'] += link(
            os.path.join(scannet_dir, f'{name}_scannet_atoms.pkl'),
            os.path.join(cache, 'scannet_embeddings', f'{name}_scannet_atoms.pkl'),
        )
        for suffix in ['_non_ligand_.ent', '_ligand.pdb']:
            counts['pdb'] += link(
                os.path.join(ligand_dir, ligand, name + suffix),
                os.path.join(cache, 'pdb_files', ligand, name + suffix),
            )
        digest = hashlib.md5(f'{name}_non_ligand_.ent'.encode()).hexdigest()
        counts['esm'] += link(
            os.path.join(ligand_dir, ESM_CACHE_NAME, f'{digest}.pt'),
            os.path.join(cache, 'esm_embeddings', ESM_CACHE_NAME, f'{digest}.pt'),
        )
    print(f'cache: {len(chains)} chains, new symlinks {counts}', flush=True)


if __name__ == '__main__':
    main()
