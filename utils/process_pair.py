import os
import numpy as np
import logging

from aligners import *
from objects import ProteinPair, Protein
from utils.constants import ResultHolder, LIGAND_DIR

logger = logging.getLogger(__name__)

def align_pair(pair_dict: dict, ligand_aligner: BaseStructureAligner, protein_aligners: list[BaseStructureAligner],
             save_transformed_models: bool = False, min_ligand_atoms: int = 3) -> None:
    ligand = pair_dict['ligand_id']
    holder = ResultHolder(pair_dict, ligand)
    logging.info(f"{pair_dict['ref_name']} {pair_dict['mov_name']}")
    if len(pair_dict['ref_chain']) != 1 or len(pair_dict['mov_chain']) != 1:
        holder.failure_message = "invalid chain given"
        return holder
    try:
        ref_protein = Protein(pair_dict["ref_name"], pair_dict["ref_chain"], ligand, save_models=save_transformed_models)
        mov_protein = Protein(pair_dict["mov_name"], pair_dict["mov_chain"], ligand, save_models=save_transformed_models)
        pair = ProteinPair(ref_protein, mov_protein, ligand, save_transformed_models=save_transformed_models)
    except Exception as e:
        holder.failure_message = str(type(e))
        return holder
    
    ref_ligand_n_atoms, mov_ligand_n_atoms = pair.number_of_ligand_atoms
    holder.ref_ligand_n_atoms = ref_ligand_n_atoms
    holder.mov_ligand_n_atoms = mov_ligand_n_atoms
                
    pair.find_ligand_transformations(holder, ligand_aligner, min_ligand_atoms=min_ligand_atoms)
    
    for aligner in protein_aligners:
        pair.find_protein_transformations(holder, aligner)

    return holder


def transformations_rmsd(mov_name: str, mov_chain: str, gt_trans: np.ndarray, aligner_trans: np.ndarray, ligand: str, ligand_res_idx: int, save_pocket: bool = True) -> None:
    mov_protein = Protein(mov_name, mov_chain, ligand, save_models=False)
    try:
        pocket_atoms = mov_protein.get_pocket_atoms(ligand_res_idx = ligand_res_idx)
        if save_pocket:
            path = f'{LIGAND_DIR}/{ligand}/{mov_name}_pocket.pdb'
            if not os.path.exists(path):
                Protein.save_coordinates_to_pdb(pocket_atoms, path)
                logging.info('saved pocket model')
        pocket_rmsd = ProteinPair.compute_rmsd(pocket_atoms, gt_trans, aligner_trans)

        ligand_residue = mov_protein.get_ligand_residues()[ligand_res_idx] # TODO: handle ligand with more residues
        ligand_atoms_coors = [atom.coord for atom in ligand_residue.get_atoms() if atom.element != "H"]
        ligand_rmsd = ProteinPair.compute_rmsd(np.vstack(ligand_atoms_coors), gt_trans, aligner_trans)
    except Exception as e:
        logging.info(e)
        return None, None
    return pocket_rmsd, ligand_rmsd