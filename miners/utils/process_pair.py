import os
import numpy as np
import logging
import pickle

from aligners import *
from objects import ProteinPair, Protein
from miners.utils.constants import BaselineHolder, ResultHolder, LIGAND_DIR

logger = logging.getLogger(__name__)

def align_pair(pair_dict: dict, ligand_aligner: BaseStructureAligner,
             save_transformed_models: bool = False, min_ligand_atoms: int = 3) -> None:
    ligand = pair_dict['ligand_id']
    holder = ResultHolder(pair_dict, ligand)
    logging.info(f"{pair_dict['ref_name']} {pair_dict['mov_name']}")
    if len(pair_dict['ref_chain']) != 1 or len(pair_dict['mov_chain']) != 1:
        holder.failure_message = "invalid chain given"
        return holder
    try:
        ref_protein = Protein(pair_dict["ref_name"], pair_dict["ref_chain"], ligand, save_models=True)
        mov_protein = Protein(pair_dict["mov_name"], pair_dict["mov_chain"], ligand, save_models=True)
        pair = ProteinPair(ref_protein, mov_protein, ligand, save_transformed_models=save_transformed_models)
    except Exception as e:
        logging.error(f"Error loading proteins: {e}")
        holder.failure_message = str(type(e))
        return holder
    
    ref_ligand_n_atoms, mov_ligand_n_atoms = pair.number_of_ligand_atoms
    holder.ref_ligand_n_atoms = ref_ligand_n_atoms
    holder.mov_ligand_n_atoms = mov_ligand_n_atoms
                
    pair.find_ligand_transformations(holder, ligand_aligner, min_ligand_atoms=min_ligand_atoms)
    print(f"finished aligning {pair_dict['ref_name']} to {pair_dict['mov_name']} for ligand {ligand}")

    return holder


def baseline_pair(pair_dict: dict, protein_aligners: list[BaseStructureAligner]) -> None:
    ligand = pair_dict['ligand_id']
    holder = BaselineHolder(pair_dict, ligand)
    logging.info(f"{pair_dict['ref_name']} {pair_dict['mov_name']}")
    if len(pair_dict['ref_chain']) != 1 or len(pair_dict['mov_chain']) != 1:
        holder.failure_message = "invalid chain given"
        return holder
    try:
        ref_protein = Protein(pair_dict["ref_name"], pair_dict["ref_chain"], ligand, save_models=True)
        mov_protein = Protein(pair_dict["mov_name"], pair_dict["mov_chain"], ligand, save_models=True)
        pair = ProteinPair(ref_protein, mov_protein, ligand, save_transformed_models=False)
    except Exception as e:
        logging.error(f"Error loading proteins: {e}")
        holder.failure_message = str(type(e))
        return holder
    
    for aligner in protein_aligners:
        pair.find_protein_transformations(holder, aligner)

    return holder


def transformations_rmsd(mov_name: str, mov_chain: str, gt_trans: np.ndarray, aligner_trans: np.ndarray, ligand: str, ligand_res_idx: int, save_pocket: bool = True) -> None:
    mov_protein = Protein(mov_name, mov_chain, ligand, save_models=False)
    try:
        pocket_atoms, _ = mov_protein.get_pocket_atoms(ligand_res_idx = ligand_res_idx)
        if save_pocket:
            path = f'{LIGAND_DIR}/{ligand}/{mov_name}_pocket.pdb'
            # if not os.path.exists(path):
            Protein.save_coordinates_to_pdb(pocket_atoms, path)
            logging.info('saved pocket model')
            return None, None
        pocket_rmsd = ProteinPair.compute_rmsd(pocket_atoms, gt_trans, aligner_trans)

        ligand_residue = mov_protein.get_ligand_residues()[ligand_res_idx] # TODO: handle ligand with more residues
        ligand_atoms_coors = [atom.coord for atom in ligand_residue.get_atoms() if atom.element != "H"]
        ligand_rmsd = ProteinPair.compute_rmsd(np.vstack(ligand_atoms_coors), gt_trans, aligner_trans)
    except Exception as e:
        logging.info(e)
        return None, None
    return pocket_rmsd, ligand_rmsd

def save_pockets(mov_name: str, mov_chain: str, ligand: str, ligand_res_idx: int, save_to_pdb=False) -> None:
    """
    Saves pocket data (coordinates and residue indices) for a given ligand-protein pair into a single file.

    Args:
        mov_name (str): Name of the protein.
        mov_chain (str): Chain identifier of the protein.
        ligand (str): Ligand identifier.
        ligand_res_idx (int): Ligand residue index.
    """
    # Define path for saving the combined data
    pocket_data_path = f'{LIGAND_DIR}/{ligand}/{mov_name}{mov_chain}_pocket_data.pkl'
    if not os.path.exists(pocket_data_path):

        # Load protein object
        mov_protein = Protein(mov_name, mov_chain, ligand, save_models=False)

        # Get the pocket atoms and residue indices
        pocket_ca_coords, pocket_ca_residue_indices = mov_protein.get_pocket_atoms(ligand_res_idx=ligand_res_idx)
  
        # Combine all pocket data into a single dictionary
        pocket_data = {
            # "all_pocket_atom_coords": pocket_atoms_coords.tolist(),
            # "pocket_ca_coords": pocket_ca_coords.tolist(),
            "residue_indices": pocket_ca_residue_indices,
        }

        try:
            # Save the data to a single pickle file
            with open(pocket_data_path, 'wb') as pocket_data_file:
                pickle.dump(pocket_data, pocket_data_file)

            logging.info('Saved pocket data (coordinates and residue indices) in a single pickle file')

        except Exception as e:
            logging.error(f"Error saving pocket data: {e}")


def save_pockets_pdb(mov_name: str, mov_chain: str, ligand: str, ligand_res_idx: int) -> None:
    """
    Saves pocket data (coordinates and residue indices) for a given ligand-protein pair into a single file.

    Args:
        mov_name (str): Name of the protein.
        mov_chain (str): Chain identifier of the protein.
        ligand (str): Ligand identifier.
        ligand_res_idx (int): Ligand residue index.
    """
    # Define path for saving the combined data
    try:
        path = f'{LIGAND_DIR}/{ligand}/{mov_name}_pocket.pdb'
        if not os.path.exists(path):
            mov_protein = Protein(mov_name, mov_chain, ligand, save_models=False)
            pocket_atoms, _ = mov_protein.get_pocket_atoms(ligand_res_idx = ligand_res_idx)
            Protein.save_coordinates_to_pdb(pocket_atoms, path)
            logging.info('saved pdb pocket model')
    except:
        logging.error(f"Error saving pocket data")