import re
import json
import logging
import os
from datetime import datetime
import multiprocessing


from utils.misc import build_object
from alligners import *
from process_pair import process_pair

current_time = datetime.now()
log_filename = current_time.strftime("%Y-%m-%d_%H-%M-%S") + ".log"
logging.basicConfig(filename=os.path.join("logs", log_filename), level=logging.INFO, format='%(message)s')

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    
    with open ('good_ligands.txt') as good_ligands_file:
        ligands =  [line.strip() for line in good_ligands_file]

    for ligand in ligands:
        logging.info(f"\nProcess ligand: {ligand}\n")

        with open(f'alligned_structures/{ligand}/config.json') as f:
            config = json.load(f)
        
        pairs = []
        with open(config['protein_names_list']) as f:
            for line in f:
                pair = {}
                proteins_and_chains = re.split(r'[:\s]', line)
                pairs.append(
                    dict(
                        ref_name=proteins_and_chains[0],
                        ref_chain=proteins_and_chains[1],
                        mov_name=proteins_and_chains[2],
                        mov_chain=proteins_and_chains[3],
                    ))
            logging.debug(f"found {len(pairs)} pairs")
        
        if len(pairs) == 0:
            continue

        alligners: list[BaseStructureAlligner] = [build_object(alligner, "alligners") for alligner in config['alligners']]
        
        manager = multiprocessing.Manager()
        result_list = manager.list()
        pool = multiprocessing.Pool()

        for pair_dict in pairs:
            pool.apply(process_pair, (pair_dict, alligners, config, result_list))

        pool.close()
        pool.join()
        positive_results = [n_trans for n_trans in list(result_list) if n_trans > 0]
        mean_result = sum(positive_results) / len(positive_results) if positive_results else 0
        logger.info(f"\nLigand had {mean_result} transformations in average\n\n")