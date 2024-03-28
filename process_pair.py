import re
import json
import logging
import os
from datetime import datetime


from alligners import *
from objects import ProteinPair, Protein


logger = logging.getLogger(__name__)

def process_pair(pair_dict: dict, alligners: list[BaseStructureAlligner], ligand_config, result_list):
    ref_protein = Protein(pair_dict["ref_name"], pair_dict["ref_chain"], ligand_config["ligand_name"], ligand_config["pdb_ligand_id"])
    mov_protein = Protein(pair_dict["mov_name"], pair_dict["mov_chain"], ligand_config["ligand_name"], ligand_config["pdb_ligand_id"])
    if ref_protein.get_num_of_ligand_atoms() < 10:
        logger.debug(f"Failed to create transformation between {pair_dict['ref_name']} and {pair_dict['mov_name']}. ligand number of atoms is less than 10 atoms, skip")
        return 
                
    pair = ProteinPair(ref_protein, mov_protein, ligand_config["ligand_name"], ligand_config["pdb_ligand_id"])
            
    for alligner in alligners:
        num_trans: int = pair.allign_atoms(alligner)
    
    if num_trans > 0:
        logger.info(f"{pair_dict['ref_name']} {pair_dict['mov_name']} had {num_trans} transformations")            
        result_list.append(num_trans)
    else:
        logger.info(f"Failed to create transformation between {pair_dict['ref_name']} and {pair_dict['mov_name']}")