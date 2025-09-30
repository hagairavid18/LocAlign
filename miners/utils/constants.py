from typing import Any
import numpy as np



# LIGAND_DIR = '/home/iscb/wolfson/hagairavid/ligands_02_01_2025'
# LIGAND_DIR = '/home/iscb/wolfson/hagairavid/ligands_10_08_2025'
LIGAND_DIR = '/home/iscb/wolfson/hagairavid/ligands_17_08_2025'
# LIGAND_DIR = '/home/iscb/wolfson/hagairavid/ligands_24_09_2025_examples'
RESULTS_COLUMNS = ['Ligand_ID', 'ref_protein', 'mov_protein', 'ref_chain', 'mov_chain',
                    'n_transformations','rotations', 'translations', 'rmse', 'coverage', 'cath_degree',
                     'ref_ligand_n_atoms', 'mov_ligand_n_atoms', "failure_message", 'p_rotations', 'p_translations', 'p_rmsd', "p_coverage", "p_message"]

NOT_ENOUGH_ATOMS_MESSAGE = "One of the ligands has less than 3 atoms"
LIGAND_RESIDUE_IS_MISSED_MESSAGE = "Could not find ligand residue for the given chain"
TOO_MUCH_RESIDUES_MESSAGE = "Too much residues to compute for a single pair"
N_ATOMS_RATIO_MESSAGE = "The lengths of the lignads differ significantly. The length ratio exceeds 1.2"
LIGAND_OVERLAP_MESSAGE = "The lignad atoms lack sufficient overlap, with less than 80% of the smaller one having corresponding atoms in the longer one."

class ResultHolder:
    def __init__(self, pair_dict: dict[str, Any], ligand: str):
        self.Ligand_ID: str = ligand
        self.ref_protein: str = pair_dict['ref_name']
        self.mov_protein: str = pair_dict['mov_name']
        self.ref_chain: str = pair_dict['ref_chain']
        self.mov_chain: str = pair_dict['mov_chain']
        self.n_transformations: int = -1
        self.rotations: list[list[np.ndarray]] = []
        self.translations: list[list[np.ndarray]] = []
        self.rmse: list[list[float]] = []
        self.coverage: list[list[float]] = []
        self.bbr: list[list[float]] = []
        self.bbc: list[list[float]] = []
        self.cath_degree: int = pair_dict['cath_level']
        self.ref_ligand_n_atoms: int = -1
        self.mov_ligand_n_atoms: int = -1
        self.n_residues_ref_ligand: int = -1
        self.n_residues_mov_ligand: int = -1
        self.failure_message: str = ""

class BaselineHolder:
    def __init__(self, pair_dict: dict[str, Any], ligand: str):
        self.Ligand_ID: str = ligand
        self.ref_protein: str = pair_dict['ref_name']
        self.mov_protein: str = pair_dict['mov_name']
        self.ref_chain: str = pair_dict['ref_chain']
        self.mov_chain: str = pair_dict['mov_chain']
        self.cath_degree: int = pair_dict['cath_level']
        self.failure_message: str = ""