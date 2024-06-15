import argparse
import json
import logging
import os
from datetime import datetime
import multiprocessing
import numpy as np
import pandas as pd

from utils.misc import save_results_to_csv
from utils.loading import deserialize_nested_lists
from alligners import *
from utils.process_pair import transformations_rmsd

start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(filename=os.path.join("logs", "baseline_" + start_time + ".log"), level=logging.INFO, format='%(message)s')

logger = logging.getLogger(__name__)


def run(pairs_df: pd.DataFrame, aligners: list[str], debug: bool = False) -> None:
     
    pool = multiprocessing.Pool()
    
    logging.info(f"\compute rmsd for: {aligners}\n")
    
    for ligand_idx, (ligand, ligand_pairs) in enumerate(pairs_df.groupby('Ligand_ID')):
        logging.info(f"\nProcess ligand: {ligand}\n")

        for row_idx, row in ligand_pairs.iterrows():
            logging.info(f"ref: {row['ref_protein']} mov: {row['mov_protein']}")

            gt_T, pred_T = np.eye(4), np.eye(4)
            for aligner in aligners:
                if len(row[f'{aligner}_rotations']) < 1:
                    pairs_df.at[row_idx, 'rmsd'] = -1
                    continue
                results_async, row_rmsd = [], []
                pred_T[:3, :3] = row[f'{aligner}_rotations']
                pred_T[:3, 3] = row[f'{aligner}_translations']

                for ligand_res_idx, residue_rotations in enumerate(row['rotations']):
                    for j, rotation in enumerate(residue_rotations):
                        # logging.info(f"ligand_res_idx: {ligand_res_idx}")
                        gt_T[:3, :3] = rotation
                        gt_T[:, 3] = row['translations'][ligand_res_idx][j]
                        
                        if not debug:
                            results_async.append(pool.apply_async(transformations_rmsd, (row['mov_protein'], row['mov_chain'], gt_T, pred_T , ligand, ligand_res_idx // row['n_residues_ref_ligand'])))
                        else:
                            row_rmsd.append(transformations_rmsd(row['mov_protein'], row['mov_chain'], gt_T, pred_T , ligand, ligand_res_idx // row['n_residues_ref_ligand']))
                if not debug:
                    row_rmsd = [result.get() for result in results_async]
                logging.info(f"{aligner} rmsd: {row_rmsd} cath: {row['cath_degree']} protein_rmsd: {row[f'{aligner}_rmsd']}")
                pairs_df.at[row_idx, f'{aligner}_rmsd'] = min(row_rmsd) if len(row_rmsd) > 0 else 0
        
    pool.close()
    pool.join()
    save_results_to_csv(pairs_df, start_time, 'baseline_results')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='RMSD parser')

    parser.add_argument('-c', '--config')
    parser.add_argument('-d', '--debug', action='store_true', help='In debug mode, multiprocess is disabled')


    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    
    pairs = pd.read_csv(config['pairs_df'])
    for col in pairs.columns:
        if pairs[col].apply(lambda x: isinstance(x, str) and x.startswith('[') and x.endswith(']')).any():
            
            pairs[col] = pairs[col].apply(lambda x: json.loads(x))
            pairs[col] = pairs[col].apply(lambda x: deserialize_nested_lists(x, col))

    run(pairs, config['protein_alligner'], args.debug)
