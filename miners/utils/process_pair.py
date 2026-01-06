import os
import logging
import pickle

from aligners import *
from objects import ProteinPair, Protein
from miners.utils.constants import BaselineHolder, ResultHolder, LIGAND_DIR

logger = logging.getLogger(__name__)


def align_pair(pair_dict: dict,
             save_transformed_models: bool = False, min_ligand_atoms: int = 3) -> None:
    ligand = pair_dict['ligand_id']
    holder = ResultHolder(pair_dict, ligand)
    logging.info(f"{pair_dict['tar_name']} {pair_dict['src_name']}")
    if len(pair_dict['tar_chain']) != 1 or len(pair_dict['src_chain']) != 1:
        holder.failure_message = "invalid chain given"
        return holder
    try:
        tar_protein = Protein(pair_dict["tar_name"], pair_dict["tar_chain"], ligand, save_models=True)
        src_protein = Protein(pair_dict["src_name"], pair_dict["src_chain"], ligand, save_models=True)
        pair = ProteinPair(tar_protein, src_protein, ligand, save_transformed_models=save_transformed_models)
    except Exception as e:
        logging.error(f"Error loading proteins: {e}")
        holder.failure_message = str(type(e))
        return holder
    
    tar_ligand_n_atoms, src_ligand_n_atoms = pair.number_of_ligand_atoms
    holder.tar_ligand_n_atoms = tar_ligand_n_atoms
    holder.src_ligand_n_atoms = src_ligand_n_atoms
                
    pair.find_ligand_transformations(holder)
    print(f"finished aligning {pair_dict['tar_name']} to {pair_dict['src_name']} for ligand {ligand}")

    return holder


def baseline_pair(pair_dict: dict, protein_aligners: list[BaseStructureAligner]) -> None:
    ligand = pair_dict['ligand_id']
    holder = BaselineHolder(pair_dict, ligand)
    logging.info(f"{pair_dict['tar_name']} {pair_dict['src_name']}")
    if len(pair_dict['tar_chain']) != 1 or len(pair_dict['src_chain']) != 1:
        holder.failure_message = "invalid chain given"
        return holder
    try:
        tar_protein = Protein(pair_dict["tar_name"], pair_dict["tar_chain"], ligand, save_models=True)
        src_protein = Protein(pair_dict["src_name"], pair_dict["src_chain"], ligand, save_models=True)
        pair = ProteinPair(tar_protein, src_protein, ligand, save_transformed_models=False)
    except Exception as e:
        logging.error(f"Error loading proteins: {e}")
        holder.failure_message = str(type(e))
        return holder
    
    for aligner in protein_aligners:
        pair.find_protein_transformations(holder, aligner)

    return holder


def save_pockets(src_name: str, src_chain: str, ligand: str, ligand_res_idx: int) -> None:
    """
    Saves pocket data (coordinates and residue indices) for a given ligand-protein pair into a single file.

    Args:
        src_name (str): Name of the protein.
        src_chain (str): Chain identifier of the protein.
        ligand (str): Ligand identifier.
        ligand_res_idx (int): Ligand residue index.
    """
    # Define path for saving the combined data
    pocket_data_path = f'{LIGAND_DIR}/{ligand}/{src_name}{src_chain}_pocket_data.pkl'
    if not os.path.exists(pocket_data_path):

        # Load protein object
        src_protein = Protein(src_name, src_chain, ligand, save_models=False)

        # Get the pocket atoms and residue indices
        pocket_ca_coords, pocket_ca_residue_indices = src_protein.get_pocket_atoms(ligand_res_idx=ligand_res_idx)
  
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
