from multiprocessing.managers import ListProxy

from alligners import *
from objects import ProteinPair, Protein

def process_pair(pair_dict: dict, alligner: BaseStructureAlligner, ligand: str, result_list: ListProxy, min_ligand_atoms: int = 3) -> None:
    if len(pair_dict['ref_chain']) != 1 or len(pair_dict['mov_chain']) != 1:
        return result_list.append((ligand, pair_dict["ref_name"], pair_dict["mov_name"], pair_dict["ref_chain"], pair_dict["mov_chain"], None, None, None, pair_dict['cath_level'] ,None, None, "invalid chain given"))


    ref_protein = Protein(pair_dict["ref_name"], pair_dict["ref_chain"], ligand, "H_" + ligand)
    mov_protein = Protein(pair_dict["mov_name"], pair_dict["mov_chain"], ligand, "H_" + ligand)
    pair = ProteinPair(ref_protein, mov_protein, ligand,  "H_" + ligand)
    
    ref_ligand_n_atmos, mov_ligand_n_atmos = pair.number_of_ligand_atoms

    if ref_ligand_n_atmos < min_ligand_atoms or mov_ligand_n_atmos < min_ligand_atoms:
        result_list.append((ligand, pair_dict["ref_name"], pair_dict["mov_name"], pair_dict["ref_chain"], pair_dict["mov_chain"], None, None, None, pair_dict['cath_level'] ,ref_ligand_n_atmos, mov_ligand_n_atmos, "Ligand too small"))
        return 
                
    num_trans, rmse, coverage, message = pair.find_transformations(alligner)
    
    result_list.append((ligand, pair_dict["ref_name"], pair_dict["mov_name"], pair_dict["ref_chain"], pair_dict["mov_chain"], num_trans, rmse, coverage, pair_dict['cath_level'] , ref_ligand_n_atmos, mov_ligand_n_atmos, message))