import argparse
import json
import logging
import os
from datetime import datetime
import multiprocessing
from typing import Any
import pandas as pd

from utils.misc import build_object
from utils.constants import RESULTS_COLUMNS
from alligners import *
from process_pair import process_pair
from parsers.utils import parse_protein_pairs

start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(filename=os.path.join("logs", start_time + ".log"), level=logging.INFO, format='%(message)s')


file_handler = logging.FileHandler(os.path.join("logs", "positive_ligands_" + start_time + ".log"))
file_handler.setLevel(logging.INFO)  # Set the log level for the file handler

logger = logging.getLogger(__name__)


def save_results_to_csv(results: list[tuple], base_dir: str = "temp_results") -> None:
    os.makedirs(base_dir, exist_ok=True)
    df = pd.DataFrame(results, columns=RESULTS_COLUMNS)
    df.to_csv(f'{base_dir}/{start_time}_{len(results)}.csv', index=False)

def run(ligands: list[str], alligner_config: dict[str, Any], debug: bool = False) -> None:
     
    alligner: BaseStructureAlligner = build_object(alligner_config, "alligners")
    manager = multiprocessing.Manager()
    result_list = manager.list()
    pool = multiprocessing.Pool()
    
    for ligand in ligands:
        logging.info(f"\nProcess ligand: {ligand}\n")

        ligand_pairs: list[dict[str, str]] = parse_protein_pairs(ligand)
        
        for pair_dict in ligand_pairs:
            if len(list(result_list)) % 100 ==0:
                save_results_to_csv(list(result_list))
                
            if not debug:
                pool.apply(process_pair, (pair_dict, alligner, ligand, result_list))
            else:
                process_pair(pair_dict, alligner, ligand, result_list)
    
    pool.close()
    pool.join()
    save_results_to_csv(list(result_list), "results")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Allignment parser')

    parser.add_argument('-c', '--config')
    parser.add_argument('-d', '--debug', action='store_true', help='In debug mode, multiprocess is disabled')


    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    
    if not config['run_single']:
        if os.path.exists(config['ligand_list']):
            with open (config['ligand_list']) as ligand_file:
                ligands =  [line.strip() for line in ligand_file]
        else:
            ligands = os.listdir("alligned_structures")
    else:
        ligands = [os.listdir("alligned_structures")[0]]

    run(ligands, config['alligner'], args.debug)
    