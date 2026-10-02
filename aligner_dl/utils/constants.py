import os
import subprocess
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

DATA_ROOT_ENV = 'LOCALIGN_DATA_ROOT'
DATA_ROOT_SENTINEL = 'ablation_dfs'


def find_data_root() -> Path:
    """Directory that holds the repo data (datasets, checkpoints, ablation_dfs, results).

    Resolution order: the LOCALIGN_DATA_ROOT environment variable; the repo root (derived
    from this file) if it contains DATA_ROOT_SENTINEL; the main checkout of the repository
    (through the git common dir) when running from a worktree and it contains
    DATA_ROOT_SENTINEL; the repo root otherwise (a fresh clone without the untracked data).
    """
    env = os.environ.get(DATA_ROOT_ENV)
    if env:
        root = Path(env).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f'{DATA_ROOT_ENV}={env} is not a directory')
        return root
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / DATA_ROOT_SENTINEL).is_dir():
        return repo_root
    result = subprocess.run(
        ['git', 'rev-parse', '--git-common-dir'],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        main_checkout = (repo_root / result.stdout.strip()).resolve().parent
        if (main_checkout / DATA_ROOT_SENTINEL).is_dir():
            return main_checkout
    return repo_root


def env_path(
    env_name: str,
    default: Path,
) -> str:
    """Path from the environment variable `env_name`, else `default`.

    The environment includes the variables loaded from DATA_ROOT/.env (when python-dotenv is
    installed); variables already set in the shell or job take precedence over the file.
    """
    return str(Path(os.environ.get(env_name) or default).expanduser())


DATA_ROOT = find_data_root()
if load_dotenv is not None:
    load_dotenv(DATA_ROOT / '.env', override=False)
PATH_CONFIG_KEYS = ('ckpt_path', 'df_path', 'base_scannet_path')
MUST_EXIST_CONFIG_KEYS = ('ckpt_path', 'df_path')

CHECKPOINTS_DIR = str(DATA_ROOT / 'checkpoints')
DATASETS_DIR = str(DATA_ROOT / 'datasets')
ABLATION_DFS_DIR = str(DATA_ROOT / 'ablation_dfs')
LIGAND_DIR = env_path('LOCALIGN_LIGAND_DIR', DATA_ROOT.parent / 'ligands_25_10_2025')
SCANNET_DIR = env_path('LOCALIGN_SCANNET_DIR', DATA_ROOT.parent / 'scannet_2212')
SCANNET_ATOM_TYPES_DIR = env_path('LOCALIGN_SCANNET_ATOM_TYPES_DIR', DATA_ROOT.parent / 'scannet_atom_types')
SOFTALIGN_DIR = env_path('LOCALIGN_SOFTALIGN_DIR', DATA_ROOT.parent / 'SoftAlign')
PLASMA_DIR = env_path('LOCALIGN_PLASMA_DIR', DATA_ROOT.parent / 'PLASMA-Protein-Local-Alignment')
DALI_DIR = env_path('LOCALIGN_DALI_DIR', DATA_ROOT.parent / 'DaliLite.v5')
ABLATION_SPLIT_DIRS = [
    str(DATA_ROOT / 'ablation_dfs' / 'homology_split'),
    str(DATA_ROOT / 'ablation_dfs' / 'ligand_split'),
]
APO_REANALYSIS_SPLIT_DIRS = [
    str(DATA_ROOT / 'ablation_dfs' / 'apo_reanalysis' / 'homology_split'),
    str(DATA_ROOT / 'ablation_dfs' / 'apo_reanalysis' / 'ligand_split'),
]
HOMOLOGY_BASELINE_CSV = str(DATA_ROOT / 'ablation_dfs' / 'homology_split' / 'baseline.csv')
MMSEQS_BINARY_NAME = 'mmseqs'
APOC_BINARY_NAME = 'apoc'
PATH_PLACEHOLDERS = {
    'SCANNET_DIR': SCANNET_DIR,
    'SCANNET_ATOM_TYPES_DIR': SCANNET_ATOM_TYPES_DIR,
    'LIGAND_DIR': LIGAND_DIR,
}


# now the invalid ligands are stored as strings, written one by one.
INVALID_LIGANDS = ['144', '15P', '1PE', '2F2', '2JC', '3HR', '3SY', '7N5', '7PE', '9JE', 'AAE', 'ABA', 'ACE', 'ACN', 'ACT', 'ACY', 'AZI', 'BAM', 'BCN', 'BCT', 'BDN', 'BEN', 'BME', 'BO3', 'BTB', 'BTC', 'BU1', 'C8E', 'CAD', 'CAQ', 'CBM', 'CCN', 'CIT', 'CL',
'CM', 'CMO', 'CO3', 'CPT', 'CXS', 'D10', 'DEP', 'DIO', 'DMS', 'DN', 'DOD', 'DOX', 'EDO', 'EEE', 'EGL', 'EOH', 'EOX', 'EPE', 'ETF', 'FCY', 'FJO', 'FLC', 'FMT', 'FW5', 'GOL', 'GSH', 'GTT', 'GYF', 'HED', 'IHP', 'IHS', 'IMD', 'IOD', 'IPA', 'IPH',
'LDA', 'MB3', 'MEG', 'MES', 'MLA', 'MLI', 'MOH', 'MPD', 'MRD', 'MSE', 'MYR', 'N', 'NA', 'NH2', 'NH4', 'NHE', 'NO3', 'O4B', 'OHE', 'OLA', 'OLC', 'OMB', 'OME', 'OXA', 'P6G', 'PE3', 'PE4', 'PEG', 'PEO', 'PEP', 'PG0', 'PG4', 'PGE', 'PGR',
'PLM', 'PO4', 'POL', 'POP', 'PVO', 'SAR', 'SCN', 'SEO', 'SEP', 'SIN', 'SO4', 'SPD', 'SPM', 'SR', 'STE', 'STO', 'STU', 'TAR', 'TBU', 'TME', 'TPO', 'TRS', 'UNK', 'UNL', 'UNX', 'UPL', 'URE']


# now with these: SO4, GOL, EDO, PO4, ACT, PEG, DMS, TRS, PGE, PG4, FMT, EPE, MPD, MES, CD, IOD
CRYSTALIZATION_LIGANDS = ['SO4', 'GOL', 'EDO', 'PO4', 'ACT', 'PEG', 'DMS', 'TRS', 'PGE', 'PG4', 'FMT', 'EPE', 'MPD', 'MES', 'CD', 'IOD']

ALL_INVALID_LIGANDS = INVALID_LIGANDS + CRYSTALIZATION_LIGANDS

RCSB_CHEMCOMP_URL = 'https://data.rcsb.org/rest/v1/core/chemcomp/{ccd_id}'
RCSB_DESCRIPTOR_KEY = 'rcsb_chem_comp_descriptor'
RCSB_SMILES_STEREO_KEYS = ('smiles_stereo', 'SMILES_stereo')
RCSB_SMILES_KEYS = ('smiles', 'SMILES')

SYMMETRY_COUNTS_PATH = str(DATA_ROOT / 'ablation_dfs' / 'ligand_symmetry_counts.json')
SYMMETRY_RELAXED_LIGAND_RMSD = 6.0
