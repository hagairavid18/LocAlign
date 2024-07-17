import argparse
import json
import logging
import os
from datetime import datetime
import multiprocessing
from typing import Any

from utils.misc import build_object, save_results_to_csv
from utils.constants import LIGAND_DIR
from alligners import *
from utils.process_pair import align_pair
from parsers.pair import parse_protein_pairs

start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(filename=os.path.join("logs", start_time + ".log"), level=logging.INFO, format='%(message)s')

logger = logging.getLogger(__name__)


def run(ligands: list[str], ligand_alligner_config: dict[str, Any], protein_alligner_config: dict[str, Any],
         debug: bool = False, save_transformed_models: bool = False) -> None:
     
    ligand_alligner: BaseStructureAlligner = build_object(ligand_alligner_config, "alligners")
    protein_alligners: list[BaseStructureAlligner | DaliAligner] = [build_object(alligner_config, "alligners") for alligner_config in protein_alligner_config]
    result_list = []
    pool = multiprocessing.Pool()
    
    for i, ligand in enumerate(ligands):
        logging.info(f"\nProcess ligand: {ligand}\n")

        ligand_pairs: list[dict[str, str]] = parse_protein_pairs(ligand)
        
        if not debug:
            results_async = [pool.apply_async(align_pair, (pair_dict, ligand_alligner, protein_alligners, ligand, save_transformed_models)) for pair_dict in ligand_pairs]
            result_list += [result.get() for result in results_async]
        else:
            result_list += [align_pair(pair_dict, ligand_alligner, protein_alligners, ligand, save_transformed_models) for pair_dict in ligand_pairs]
        if i % 2 == 0:
            save_results_to_csv(result_list, start_time, "alignment_temp_results")
    
    pool.close()
    pool.join()
    save_results_to_csv(result_list, start_time, "alignment_results")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Allignment parser')
    parser.add_argument('-c', '--config')
    parser.add_argument('-d', '--debug', action='store_true', help='In debug mode, multiprocess is disabled')

    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    
    if config['ligand']:

        ligands =  [config['ligand']]
    else:
        ligands = os.listdir(LIGAND_DIR)

    run(ligands, config['ligand_alligner'], config['protein_alligner'], args.debug, config['save_transformed_models'])
