import argparse
import json
import logging
import os
from datetime import datetime
import multiprocessing
from typing import Any
import pandas as pd

from miners.utils.misc import build_object, save_results_to_csv
from miners.utils.process_pair import baseline_pair
from parsers.pair import parse_protein_pairs

CHUNK_SIZE = 50

start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(os.path.join("miners", "logs", 'baseline'), exist_ok=True)
logging.basicConfig(filename=os.path.join("miners", "logs", 'baseline', start_time + ".log"), level=logging.INFO, format='%(message)s')

logger = logging.getLogger(__name__)


def run(pairs_df: pd.DataFrame, protein_aligner_config: dict[str, Any], save_path: str,
         debug: bool = False, prev_df: pd.DataFrame | None = None) -> None:
     
    protein_aligners = [build_object(aligner_config, "aligners") for aligner_config in protein_aligner_config]
    result_list = []
    pool = multiprocessing.Pool(30)
    
    ligand_pairs: list[dict[str, str]] = parse_protein_pairs(pairs_df, prev_df)

    for i in range(0, len(ligand_pairs), CHUNK_SIZE):
        chunk = ligand_pairs[i:i + CHUNK_SIZE]
        if not debug:
            results_async = [pool.apply_async(baseline_pair, (pair_dict, protein_aligners)) for pair_dict in chunk]
            result_list += [result.get() for result in results_async]
        else:
            result_list += [baseline_pair(pair_dict, protein_aligners) for pair_dict in chunk]
        save_results_to_csv(result_list, start_time, "baseline_temp_results_sw", prev_df)
    
    pool.close()
    pool.join()
    df = save_results_to_csv(result_list, start_time, "baseline_results_sw", prev_df)
    # pairs_df = pairs_df.drop(['TMAligner_protein_rmsd', 'TMAligner_rmsd', 'TMAligner_rotations', 'TMAligner_translations'], axis=1)
    merged_df = pd.merge(pairs_df, df, on=['ligand_id', 'tar_protein', 'tar_chain', 'src_protein', 'src_chain', 'cath_degree'], how='left')
    merged_df.to_csv(save_path, index=False)
    logging.info(f"Saved results to {save_path} {len(merged_df)} rows.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='alignment parser')
    parser.add_argument('-c', '--config')
    parser.add_argument('-d', '--debug', action='store_true', help='In debug mode, multiprocess is disabled')
    parser.add_argument('-p', '--save_path', type=str, default=None, help='Path to save the results CSV file')

    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    
    pairs_df = pd.read_csv(config['df_path'])

    if config['prev_df_path']:
        prev_df = pd.read_csv(config['prev_df_path'])
    else:
        prev_df = None

    run(pairs_df, config['protein_aligner'], args.save_path, args.debug, prev_df)
