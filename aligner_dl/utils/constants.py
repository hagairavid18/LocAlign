import os
import subprocess
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

CHECKOUT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT_ENV = 'LOCALIGN_DATA_ROOT'
DATA_ROOT_SENTINEL = 'ablation_dfs'


def find_data_root() -> Path:
    """Directory that holds the untracked repo data (checkpoints, ablation_dfs, results, ...).

    Tracked files hang off CHECKOUT_ROOT instead. Resolution order: the LOCALIGN_DATA_ROOT
    environment variable; CHECKOUT_ROOT if it contains DATA_ROOT_SENTINEL; the main checkout
    of the repository (through the git common dir, or the worktree's .git file when git is
    not on PATH) when running from a worktree and it
    contains DATA_ROOT_SENTINEL; CHECKOUT_ROOT otherwise (a fresh clone without the untracked
    data).
    """
    env = os.environ.get(DATA_ROOT_ENV)
    if env:
        root = Path(env).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f'{DATA_ROOT_ENV}={env} is not a directory')
        return root
    if (CHECKOUT_ROOT / DATA_ROOT_SENTINEL).is_dir():
        return CHECKOUT_ROOT
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--git-common-dir'],
            cwd=CHECKOUT_ROOT,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        result = None
    git_file = CHECKOUT_ROOT / '.git'
    if result is not None and result.returncode == 0:
        main_checkout = (CHECKOUT_ROOT / result.stdout.strip()).resolve().parent
    elif git_file.is_file() and git_file.read_text().startswith('gitdir:'):
        main_checkout = Path(git_file.read_text().split(':', 1)[1].strip()).resolve().parents[2]
    else:
        main_checkout = None
    if main_checkout is not None:
        if (main_checkout / DATA_ROOT_SENTINEL).is_dir():
            return main_checkout
    return CHECKOUT_ROOT


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
UNTRACKED_PATH_CONFIG_KEYS = ('ckpt_path',)
MUST_EXIST_CONFIG_KEYS = ('ckpt_path', 'df_path')

CHECKPOINTS_DIR = str(DATA_ROOT / 'checkpoints')
DATASETS_DIR = str(CHECKOUT_ROOT / 'datasets')
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

ESM_CACHE_NAME = 'esm_cache_esm2_t30_150M_UR50D_18'
BIOLIP_NR_DB_PATH = str(CHECKOUT_ROOT / 'example_inputs' / 'biolip2_nr_database.csv')
PLRMSD_AUC_DIR = str(DATA_ROOT / 'results' / 'plrmsd_auc')
PAIR_KEY_COLUMNS = ['tar_protein', 'tar_chain', 'src_protein', 'src_chain']
INFERENCE_RAW_COLUMNS = {
    '_embedding': 'embedding_similarity',
    '_gap': 'entropy',
    '_corr_rmsd': 'corr_rmsd',
    '_radius': 'radius_of_gyration',
}
PLRMSD_FEATURES = ['normalized_embedding_similarity', 'corr_rmsd', 'radius_of_gyration_ang', 'perplexity']
PLRMSD_MONOTONIC = [-1, 1, 1, -1]
PLRMSD_TARGET = 'ligand_rmsd'
PLRMSD_TARGET_CUTOFF = 10.0
PLRMSD_CAP_THRESHOLD = 9.99
ELRMSD_QUANTILE = 0.2
AUC_SCORE_DECIMALS = 3
AUC_BOOTSTRAP_RESAMPLES = 2000
AUC_LRMSD_SUCCESS_CUTOFF = 4.0
PLRMSD_POSITIVES_CSV = 'positives_cv_predictions_{split}.csv'
PLRMSD_NEGATIVES_CSV = 'negatives_{split}.csv'
PLRMSD_FAILED_NEGATIVES_CSV = 'failed_negatives_{split}.csv'
PLRMSD_CALIBRATION_PKL = 'calibration_model_{split}.pkl'
ELRMSD_CALIBRATION_PKL = 'ligand_calibration_model_{split}.pkl'
PLRMSD_BEST_PARAMS_JSON = 'best_params_{split}.json'
PLRMSD_AUC_CSV = 'auc.csv'
PLRMSD_BREAKDOWN_CSV = 'breakdown.csv'
PLRMSD_ROC_PNG = 'roc_{split}.png'
PLRMSD_NEGATIVES_INPUT_CSV = 'negatives_input_{split}.csv'
