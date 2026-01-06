import argparse
import json
import logging
import os
from datetime import datetime
import multiprocessing as mp
import sys
from typing import Any
import pandas as pd


sys.path.append(os.path.join(os.path.dirname(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from miners.utils.misc import build_object, save_results_to_csv
from aligners import *
from miners.utils.process_pair import align_pair
from parsers.pair import parse_protein_pairs

# multiprocessing.set_start_method("spawn")

CHUNK_SIZE = 5000

start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(os.path.join("miners", "logs", 'align'), exist_ok=True)
logging.basicConfig(filename=os.path.join("miners", "logs", 'align', start_time + ".log"), level=logging.INFO, format='%(message)s')

logger = logging.getLogger(__name__)


def run(pairs_df: pd.DataFrame,
         debug: bool = False, save_transformed_models: bool = False, prev_df: pd.DataFrame | None = None) -> None:
     
    result_list = []
    pool = mp.Pool(mp.cpu_count())
    
    ligand_pairs: list[dict[str, str]] = parse_protein_pairs(pairs_df, prev_df)

    for i in range(0, len(ligand_pairs), CHUNK_SIZE):
        chunk = ligand_pairs[i:i + CHUNK_SIZE]
        if not debug:
            results_async = [pool.apply_async(align_pair, (pair_dict, save_transformed_models)) for pair_dict in chunk]
            result_list += [result.get() for result in results_async]
        else:
            result_list += [align_pair(pair_dict, save_transformed_models) for pair_dict in chunk]
        save_results_to_csv(result_list, start_time, "alignment_temp_results", prev_df)
    
    pool.close()
    pool.join()
    save_results_to_csv(result_list, start_time, "alignment_results", prev_df)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description='alignment parser')
    parser.add_argument('-c', '--config')
    parser.add_argument('-d', '--debug', action='store_true', help='In debug mode, multiprocess is disabled')

    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    
    pairs_df = pd.read_csv(config['df_path'], index_col=0)
    # pairs_df = pairs_df.sample(frac=0.005, random_state=42).reset_index(drop=True)  # shuffle
    
    if config['ligand']:
        pairs_df = pairs_df[pairs_df['ligand_id'] == config['ligand']]
    
    if config['prev_df_path']:
        prev_df = pd.read_csv(config['prev_df_path'])
    else:
        prev_df = None

    run(pairs_df, args.debug, config['save_transformed_models'], prev_df)
