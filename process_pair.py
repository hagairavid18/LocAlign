import logging
from multiprocessing.managers import ListProxy

from alligners import *
from objects import ProteinPair, Protein

# from main import logger2

logger = logging.getLogger(__name__)

def process_pair(pair_dict: dict, alligner: BaseStructureAlligner, ligand: str, result_list: ListProxy) -> None:
    ref_protein = Protein(pair_dict["ref_name"], pair_dict["ref_chain"], ligand, "H_" + ligand)
    mov_protein = Protein(pair_dict["mov_name"], pair_dict["mov_chain"], ligand, "H_" + ligand)
    
    if ref_protein.get_num_of_ligand_atoms() < 10:
        logger.info(f"Failed to create transformation between {pair_dict['ref_name']} and {pair_dict['mov_name']}. ligand number of atoms is less than 10 atoms, skip")
        return 
                
    pair = ProteinPair(ref_protein, mov_protein, ligand,  "H_" + ligand)
            
    num_trans: int = pair.allign_atoms(alligner)
    
    if num_trans > 0:
        str_pair = f"{pair_dict['ref_name']}:{pair_dict['ref_chain']} {pair_dict['mov_name']}:{pair_dict['mov_chain']}"
        logger.info(f"{str_pair} had {num_trans} transformations")            
        result_list.append((str_pair ,num_trans))
        # logger2.info(f"{pair_dict["ref_name"]}:{pair_dict["ref_chain"]} {pair_dict["mov_name"]}:{pair_dict["mov_chain"]}")
    else:
        logger.info(f"Failed to create transformation between {pair_dict['ref_name']} and {pair_dict['mov_name']} no transformations")