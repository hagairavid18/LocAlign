import argparse
from datetime import datetime
import json
import logging
import os
import pandas as pd

from utils.loading import deserialize_nested_lists
from utils.process_pair import save_pockets

# Setup logging
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join("logs", "baseline")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, start_time + ".log"), level=logging.INFO, format='%(message)s')

logger = logging.getLogger(__name__)

def run(pairs_df: pd.DataFrame, debug: bool = False) -> None:
    """
    Iterates over each distinct ligand and protein and saves the pocket coordinates.
    """
    logging.info("Iterating over ligands and proteins to save pockets...")

    for ligand_idx, (ligand, ligand_pairs) in enumerate(pairs_df.groupby('Ligand_ID')):
        logging.info(f"\nProcess ligand: {ligand}\n")

        for row_idx, row in ligand_pairs.iterrows():
            logging.info(f"Processing: ref: {row['ref_protein']} mov: {row['mov_protein']}")

            try:
                # Save pocket for each protein
                save_pockets(
                    row['mov_protein'],
                    row['mov_chain'],
                    ligand=ligand,
                    ligand_res_idx=0,  # Or any other logic for ligand residue index
                )
                # Save pocket for each protein
                save_pockets(
                    row['ref_protein'],
                    row['ref_chain'],
                    ligand=ligand,
                    ligand_res_idx=0,  # Or any other logic for ligand residue index
                )
            except Exception as e:
                logging.error(f"Error processing ligand {ligand}, protein {row['mov_protein']}: {e}")
                
    logging.info("Pocket saving completed!")

def main(config_file: str, debug: bool = False):
    """
    Main function to load the config, read the pairs, and run the process.
    """
    with open(config_file) as f:
        config = json.load(f)

    # Load the pairs data
    pairs = pd.read_csv(config['pairs_df'])
    for col in pairs.columns:
        if pairs[col].apply(lambda x: isinstance(x, str) and x.startswith('[') and x.endswith(']')).any():
            pairs[col] = pairs[col].fillna('[]')
            pairs[col] = pairs[col].apply(lambda x: json.loads(x))
            pairs[col] = pairs[col].apply(lambda x: deserialize_nested_lists(x, col))

    # Run the simplified process
    run(pairs, debug)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pocket Saver')

    # Arguments
    parser.add_argument('-c', '--config', required=True, help='Path to config file')
    parser.add_argument('-d', '--debug', action='store_true', help='Run in debug mode')

    args = parser.parse_args()

    # Run main function
    main(args.config, args.debug)
