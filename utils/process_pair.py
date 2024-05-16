from multiprocessing.managers import ListProxy

from alligners import *
from objects import ProteinPair, Protein
import logging

logger = logging.getLogger(__name__)

def process_pair(pair_dict: dict, alligner: BaseStructureAlligner, ligand: str, save_transformed_ligand: bool = False, min_ligand_atoms: int = 3) -> None:
    logging.info(f"{pair_dict['ref_name']} {pair_dict['mov_name']}")
    if len(pair_dict['ref_chain']) != 1 or len(pair_dict['mov_chain']) != 1:
        return  (ligand, pair_dict["ref_name"], pair_dict["mov_name"], pair_dict["ref_chain"], pair_dict["mov_chain"], 0.0, (), {},  None, None, pair_dict['cath_level'] ,None, None, "invalid chain given")

    try:
        ref_protein = Protein(pair_dict["ref_name"], pair_dict["ref_chain"], ligand)
        mov_protein = Protein(pair_dict["mov_name"], pair_dict["mov_chain"], ligand)
        pair = ProteinPair(ref_protein, mov_protein, ligand,  "H_" + ligand, save_transformed_protein=save_transformed_ligand)
    except Exception as e:
        return  (ligand, pair_dict["ref_name"], pair_dict["mov_name"], pair_dict["ref_chain"], pair_dict["mov_chain"], 0.0, (), {},  None, None, pair_dict['cath_level'] ,None, None, str(type(e)))
    
    ref_ligand_n_atmos, mov_ligand_n_atmos = pair.number_of_ligand_atoms
                
    rotations, translations, rmse, coverage, message = pair.find_transformations(alligner, save_transformed_ligand, min_ligand_atoms=min_ligand_atoms)
    
    return (ligand, pair_dict["ref_name"], pair_dict["mov_name"], pair_dict["ref_chain"], pair_dict["mov_chain"], sum([len(rot) for rot in rotations]), rotations, translations, rmse, coverage, pair_dict['cath_level'] , ref_ligand_n_atmos, mov_ligand_n_atmos, message)