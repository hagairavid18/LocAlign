import logging
import pandas as pd

logger = logging.getLogger(__name__)


def parse_protein_pairs(pairs_df: pd.DataFrame) -> list[dict[str, str]]:
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
    return pairs