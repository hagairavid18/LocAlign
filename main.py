import argparse
import json
import logging
import os
from datetime import datetime
import multiprocessing
from typing import Any
import pandas as pd

from utils.misc import build_object
from alligners import *
from utils.process_pair import process_pair
from parsers.utils import parse_protein_pairs

start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(filename=os.path.join("logs", start_time + ".log"), level=logging.INFO, format='%(message)s')


logger = logging.getLogger(__name__)


def save_results_to_csv(results: list[tuple], base_dir: str = "temp_results") -> None:
    os.makedirs(base_dir, exist_ok=True)
    df = pd.DataFrame([obj.__dict__ for obj in results])
    df.to_csv(f'{base_dir}/{start_time}_{len(results)}.csv', index=False)

def run(ligands: list[str], ligand_alligner_config: dict[str, Any], protein_alligner_config: dict[str, Any], debug: bool = False, save_transformed_ligand: bool = False) -> None:
     
    ligand_alligner: BaseStructureAlligner = build_object(ligand_alligner_config, "alligners")
    protein_alligner: BaseStructureAlligner = build_object(protein_alligner_config, "alligners")
    result_list = []
    pool = multiprocessing.Pool()
    
    for i, ligand in enumerate(ligands):
        logging.info(f"\nProcess ligand: {ligand}\n")

        ligand_pairs: list[dict[str, str]] = parse_protein_pairs(ligand)
        
        if not debug and not protein_alligner:
            results_async = [pool.apply_async(process_pair, (pair_dict, ligand_alligner, protein_alligner, ligand, save_transformed_ligand)) for pair_dict in ligand_pairs]
            result_list += [result.get() for result in results_async]
        else:
            result_list += [process_pair(pair_dict, ligand_alligner, protein_alligner, ligand, save_transformed_ligand) for pair_dict in ligand_pairs]
        if i % 2 == 0:
            save_results_to_csv(result_list)
    
    pool.close()
    pool.join()
    save_results_to_csv(result_list, "results")


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
        ligands = os.listdir("alligned_structures")

    run(ligands, config['ligand_alligner'], config['protein_alligner'], args.debug, config['save_transformed_ligand'])
