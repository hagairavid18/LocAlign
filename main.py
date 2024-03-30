import argparse
import json
import logging
import os
from datetime import datetime
import multiprocessing
from typing import Any


from utils.misc import build_object
from alligners import *
from process_pair import process_pair
from parsers.utils import parse_protein_pairs

current_time = datetime.now()
log_filename = current_time.strftime("%Y-%m-%d_%H-%M-%S") + ".log"
logging.basicConfig(filename=os.path.join("logs", log_filename), level=logging.INFO, format='%(message)s')

logger2 = logging.getLogger('my_second_logger')
logger2.setLevel(logging.INFO)  # Set the log level for the second logger

file_handler = logging.FileHandler(os.path.join("logs", "positive_ligands_" + log_filename))
file_handler.setLevel(logging.INFO)  # Set the log level for the file handler
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger2.addHandler(file_handler)

logger = logging.getLogger(__name__)



def run(ligands: list[str], alligner_config: dict[str, Any], debug: bool = False) -> None:
     
     for ligand in ligands:
        logging.info(f"\nProcess ligand: {ligand}\n")

        pairs = parse_protein_pairs(ligand)
        
        if len(pairs) == 0:
            continue

        alligner: BaseStructureAlligner = build_object(alligner_config, "alligners")
        
        manager = multiprocessing.Manager()
        result_list = manager.list()
        pool = multiprocessing.Pool()

        for pair_dict in pairs:
            if not debug:
                pool.apply(process_pair, (pair_dict, alligner, ligand, result_list))
            else:
                process_pair(pair_dict, alligner, ligand, result_list)

        pool.close()
        pool.join()
        positive_results = [n_trans for n_trans in list(result_list) if n_trans > 0]
        mean_result = sum(positive_results) / len(positive_results) if positive_results else 0
        logger.info(f"\nLigand had {mean_result} transformations in average\n\n")

        if len(positive_results) > 0:
            logger2.info(ligand)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Allignment parser')

    parser.add_argument('-c', '--config')
    parser.add_argument('-d', '--debug', action='store_true', help='In debug mode, multiprocess is disabled')


    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    
    if not config['run_single']:
        with open (config['ligand_list']) as ligand_file:
            ligands =  [line.strip() for line in ligand_file]
    else:
        ligands = [os.listdir("alligned_structures")[0]]

    run(ligands, config['alligner'], args.debug)
    