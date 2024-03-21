import re
import json
import logging

from utils.misc import build_object
from alligners import *
from objects import ProteinPair, Protein

logging.basicConfig(level=logging.INFO, format='%(message)s')

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    
    with open('alligned_structures/ATP/config.json') as f:
        config = json.load(f)
    
    pairs = []
    with open(config['protein_names_list']) as f:
        for line in f:
            pair = {}
            proteins_and_chains = re.split(r'[:\s]', line)
            # assert len(proteins_and_chains) == 5, 'Must have format of "pdb_name1:chain_id1 pdb_name2:chain_id2'
            pairs.append(
                dict(
                    ref_name=proteins_and_chains[0],
                    ref_chain=proteins_and_chains[1],
                    mov_name=proteins_and_chains[2],
                    mov_chain=proteins_and_chains[3],
                ))

    alligners: list[BaseStructureAlligner] = [build_object(alligner, "alligners") for alligner in config['alligners']]
        
    for pair_dict in pairs:
        ref_protein = Protein(pair_dict["ref_name"], pair_dict["ref_chain"], config["ligand_name"], config["pdb_ligand_id"])
        mov_protein = Protein(pair_dict["mov_name"], pair_dict["mov_chain"], config["ligand_name"], config["pdb_ligand_id"])
        pair = ProteinPair(ref_protein, mov_protein, config["ligand_name"], config["pdb_ligand_id"])
    
        for alligner in alligners:
            logging.info(f'{alligner.name}')
            pair.allign_atoms(alligner)