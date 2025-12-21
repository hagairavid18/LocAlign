from typing import Any
from dataclasses import dataclass


@dataclass
class PairHolder:
    tar_protein: str
    tar_chain: str
    src_protein: str
    src_chain: str
    ligand: str | None = None
    tar_ligand: str | None = None
    src_ligand: str | None = None
    cath_degree: int = -1
    ligand_rmsd: float = 0.0
    tar_motif: list[int] = None
    src_motif: list[int] = None
    message: str | None = None

    def __post_init__(self) -> None:
        # Default to the shared ligand if per-chain ligands are not provided
        if self.tar_ligand is None:
            self.tar_ligand = self.ligand
        if self.src_ligand is None:
            self.src_ligand = self.ligand
        # Preserve a single ligand value for backward compatibility and logging
        if self.ligand is None:
            self.ligand = self.tar_ligand or self.src_ligand

    def to_dict(self) -> dict:
        return {
            'tar_protein': self.tar_protein,
            'tar_chain': self.tar_chain,
            'tar_motif': self.tar_motif,
            'src_protein': self.src_protein,
            'src_chain': self.src_chain,
            'src_motif': self.src_motif,
            'cath_degree': self.cath_degree,
            'Ligand RMSD': self.ligand_rmsd,
            'ligand': self.ligand,
            'tar_ligand': self.tar_ligand,
            'src_ligand': self.src_ligand,
        }


# LIGAND_DIR = '/home/iscb/wolfson/hagairavid/ligands_02_01_2025'
# LIGAND_DIR = '/home/iscb/wolfson/hagairavid/ligands_10_08_2025'
# LIGAND_DIR = '/home/iscb/wolfson/hagairavid/ligands_17_08_2025'
LIGAND_DIR = '/home/iscb/wolfson/hagairavid/ligands_25_10_2025'
# LIGAND_DIR = '/home/iscb/wolfson/hagairavid/ligands_24_09_2025_examples'
RESULTS_COLUMNS = ['Ligand_ID', 'tar_protein', 'src_protein', 'tar_chain', 'src_chain', 'cath_degree',
                     'tar_ligand_n_atoms', 'src_ligand_n_atoms', "failure_message"]

NOT_ENOUGH_ATOMS_MESSAGE = "One of the ligands has less than 3 atoms"
LIGAND_RESIDUE_IS_MISSED_MESSAGE = "Could not find ligand residue for the given chain"
TOO_MUCH_RESIDUES_MESSAGE = "Too much residues to compute for a single pair"
N_ATOMS_RATIO_MESSAGE = "The lengths of the lignads differ significantly. The length ratio exceeds 1.2"
LIGAND_OVERLAP_MESSAGE = "The lignad atoms lack sufficient overlap, with less than 80% of the smaller one having corresponding atoms in the longer one."

class ResultHolder:
    def __init__(self, pair_dict: dict[str, Any], ligand: str):
        self.Ligand_ID: str = ligand
        self.tar_protein: str = pair_dict['tar_name']
        self.src_protein: str = pair_dict['src_name']
        self.tar_chain: str = pair_dict['tar_chain']
        self.src_chain: str = pair_dict['src_chain']
        self.cath_degree: int = pair_dict['cath_level']
        self.tar_ligand_n_atoms: int = -1
        self.src_ligand_n_atoms: int = -1
        self.n_residues_tar_ligand: int = -1
        self.n_residues_src_ligand: int = -1
        self.failure_message: str = ""

class BaselineHolder:
    def __init__(self, pair_dict: dict[str, Any], ligand: str):
        self.Ligand_ID: str = ligand
        self.tar_protein: str = pair_dict['tar_name']
        self.src_protein: str = pair_dict['src_name']
        self.tar_chain: str = pair_dict['tar_chain']
        self.src_chain: str = pair_dict['src_chain']
        self.cath_degree: int = pair_dict['cath_level']
        self.failure_message: str = ""