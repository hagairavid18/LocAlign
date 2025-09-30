import logging
import pandas as pd

logger = logging.getLogger(__name__)


def parse_protein_pairs(pairs_df: pd.DataFrame, prev_df: pd.DataFrame | None = None) -> list[dict[str, str]]:
    pairs = []
    for _, line in pairs_df.iterrows():
        pairs.append(
            dict(
                ligand_id = line['ligand_id'],
                ref_name = line['ref_protein'],
                ref_chain = line['ref_chain'],
                mov_name = line['mov_protein'],
                mov_chain = line['mov_chain'],
                cath_level = line['CATH_degree'],
            ))
    # logging.info(f"found {len(pairs)} pairs for {ligand}")
    if prev_df is None:
        return pairs
    prev_pairs =  []
    for _, line in prev_df.iterrows():
        prev_pairs.append(
            dict(
                ligand_id = line['Ligand_ID'],
                ref_name = line['ref_protein'],
                ref_chain = line['ref_chain'],
                mov_name = line['mov_protein'],
                mov_chain = line['mov_chain'],
                cath_level = line['cath_degree'],
            ))
    new_pairs = [pair for pair in pairs if pair not in prev_pairs]
    print(f"found {len(new_pairs)} new pairs, {len(prev_pairs)} pairs already aligned")
    return new_pairs