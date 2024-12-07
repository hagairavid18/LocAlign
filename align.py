import argparse
import json
import logging
import os
from datetime import datetime
import multiprocessing
from typing import Any
import pandas as pd

from utils.misc import build_object, save_results_to_csv
from utils.constants import LIGAND_DIR
from aligners import *
from utils.process_pair import align_pair
from parsers.pair import parse_protein_pairs

start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(filename=os.path.join("logs", start_time + ".log"), level=logging.INFO, format='%(message)s')

logger = logging.getLogger(__name__)


def run(pairs_df: pd.DataFrame, ligand_aligner_config: dict[str, Any], protein_aligner_config: dict[str, Any],
         debug: bool = False, save_transformed_models: bool = False) -> None:
     
    ligand_aligner = build_object(ligand_aligner_config, "aligners")
    protein_aligners = [build_object(aligner_config, "aligners") for aligner_config in protein_aligner_config]
    result_list = []
    pool = multiprocessing.Pool(30)
    
    ligand_pairs: list[dict[str, str]] = parse_protein_pairs(pairs_df)

    for i in range(0, len(ligand_pairs), 500):
        chunk = ligand_pairs[i:i + 500]
        if not debug:
            results_async = [pool.apply_async(align_pair, (pair_dict, ligand_aligner, protein_aligners, save_transformed_models)) for pair_dict in chunk]
            result_list += [result.get() for result in results_async]
        else:
            result_list += [align_pair(pair_dict, ligand_aligner, protein_aligners, save_transformed_models) for pair_dict in chunk]
        save_results_to_csv(result_list, start_time, "alignment_temp_results")
    
    pool.close()
    pool.join()
    save_results_to_csv(result_list, start_time, "alignment_results")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='alignment parser')
    parser.add_argument('-c', '--config')
    parser.add_argument('-d', '--debug', action='store_true', help='In debug mode, multiprocess is disabled')

    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    
    pairs_df = pd.read_csv(config['df_path'], index_col=0)
    
    if config['ligand']:
        pairs_df = pairs_df[pairs_df['ligand_id'] == config['ligand']]

    run(pairs_df, config['ligand_aligner'], config['protein_aligner'], args.debug, config['save_transformed_models'])
