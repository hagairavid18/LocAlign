import json
import logging

from utils.misc import build_object
from alligners import *
from objects import ProteinPair, Protein

logging.basicConfig(level=logging.INFO, format='%(message)s')

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    
    with open('config.json') as f:
        config = json.load(f)
    
    pairs = []
    with open(config['protein_names_list']) as f:
        for line in f:
            pair = line.strip().split()
            pairs.append(pair)

    alligners: list[BaseStructureAlligner] = [build_object(alligner, "alligners") for alligner in config['alligners']]
        
    for pair in pairs:
        
        ref_protein = Protein(pair[0], config["ligand_name"])
        mov_protein = Protein(pair[1], config["ligand_name"])
        pair = ProteinPair(ref_protein, mov_protein, config["ligand_name"], config["pdb_ligand_id"])
    
        for alligner in alligners:
            logging.info(f'{alligner.name}')
            pair.allign_atoms(alligner)