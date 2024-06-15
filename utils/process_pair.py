import numpy as np
import logging

from alligners import *
from objects import ProteinPair, Protein
from utils.constants import ResultHolder

logger = logging.getLogger(__name__)

def align_pair(pair_dict: dict, ligand_alligner: BaseStructureAlligner, protein_alligners: list[BaseStructureAlligner],
                 ligand: str, save_transformed_ligand: bool = False, min_ligand_atoms: int = 3) -> None:
    holder = ResultHolder(pair_dict, ligand)
    logging.info(f"{pair_dict['ref_name']} {pair_dict['mov_name']}")
    if len(pair_dict['ref_chain']) != 1 or len(pair_dict['mov_chain']) != 1:
        holder.failure_message = "invalid chain given"
        return holder
    try:
        ref_protein = Protein(pair_dict["ref_name"], pair_dict["ref_chain"], ligand)
        mov_protein = Protein(pair_dict["mov_name"], pair_dict["mov_chain"], ligand)
        pair = ProteinPair(ref_protein, mov_protein, ligand, save_transformed_protein=save_transformed_ligand)
    except Exception as e:
        holder.failure_message = str(type(e))
        return holder
    
    ref_ligand_n_atoms, mov_ligand_n_atoms = pair.number_of_ligand_atoms
    holder.ref_ligand_n_atoms = ref_ligand_n_atoms
    holder.mov_ligand_n_atoms = mov_ligand_n_atoms
                
    pair.find_ligand_transformations(holder, ligand_alligner, save_transformed_ligand, min_ligand_atoms=min_ligand_atoms)
    
    for aligner in protein_alligners:
        pair.find_protein_transformations(holder, aligner, save_transformed_ligand)

    return holder


def transformations_rmsd(mov_name: str, mov_chain: str, gt_trans: np.ndarray, aligner_trans: np.ndarray, ligand: str, ligand_res_idx: int) -> None:
    mov_protein = Protein(mov_name, mov_chain, ligand)
    pocket_atoms = mov_protein.get_pocket_atoms(ligand_res_idx = ligand_res_idx)
    rmsd = ProteinPair.compute_rmsd(pocket_atoms, gt_trans, aligner_trans)
    return rmsd