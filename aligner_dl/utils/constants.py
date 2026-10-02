import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from path_roots import (  # noqa: E402
    DATA_ROOT_ENV,
    EXTERNAL_ROOT_ENV,
    find_data_root,
    find_external_root,
    find_repo_root,
)

REPO_ROOT = find_repo_root()
DATA_ROOT = find_data_root(REPO_ROOT)
EXTERNAL_ROOT = find_external_root(DATA_ROOT)
EXTERNAL_PATH_PREFIX = 'external/'
PATH_CONFIG_KEYS = ('ckpt_path', 'df_path', 'base_scannet_path')
MUST_EXIST_CONFIG_KEYS = ('ckpt_path', 'df_path')

CHECKPOINTS_DIR = str(DATA_ROOT / 'checkpoints')
DATASETS_DIR = str(DATA_ROOT / 'datasets')
ABLATION_DFS_DIR = str(DATA_ROOT / 'ablation_dfs')
LIGAND_DIR = str(EXTERNAL_ROOT / 'ligands_25_10_2025')
SCANNET_DIR_NAME = 'scannet_2212'
SCANNET_DIR = str(EXTERNAL_ROOT / SCANNET_DIR_NAME)
SOFTALIGN_DIR = str(EXTERNAL_ROOT / 'SoftAlign')
PLASMA_DIR = str(EXTERNAL_ROOT / 'PLASMA-Protein-Local-Alignment')
ABLATION_SPLIT_DIRS = [
    str(DATA_ROOT / 'ablation_dfs' / 'homology_split'),
    str(DATA_ROOT / 'ablation_dfs' / 'ligand_split'),
]
APO_REANALYSIS_SPLIT_DIRS = [
    str(DATA_ROOT / 'ablation_dfs' / 'apo_reanalysis' / 'homology_split'),
    str(DATA_ROOT / 'ablation_dfs' / 'apo_reanalysis' / 'ligand_split'),
]
HOMOLOGY_BASELINE_CSV = str(DATA_ROOT / 'ablation_dfs' / 'homology_split' / 'baseline.csv')
DALI_DIR = str(EXTERNAL_ROOT / 'DaliLite.v5')
MMSEQS_BINARY_NAME = 'mmseqs'
APOC_BINARY_NAME = 'apoc'


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
