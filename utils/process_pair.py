
from alligners import *
from objects import ProteinPair, Protein
import logging

from utils.constants import ResultHolder

logger = logging.getLogger(__name__)

def process_pair(pair_dict: dict, ligand_alligner: BaseStructureAlligner, protein_alligner: BaseStructureAlligner,
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
    
    if protein_alligner:
        pair.find_protein_transformations(holder, protein_alligner, save_transformed_ligand)

    return holder