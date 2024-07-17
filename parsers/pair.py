import logging
import os 
import re
from utils.constants import LIGAND_DIR


logger = logging.getLogger(__name__)

def parse_protein_pairs(ligand: str) -> list[dict[str, str]]:
    pairs = []
    with open(os.path.join(LIGAND_DIR, ligand, ligand + '.txt')) as f:
        for line in f:
            proteins_and_chains = re.split(r'[:\s]', line)
            pairs.append(
                dict(
                    ref_name=proteins_and_chains[0],
                    ref_chain=proteins_and_chains[1],
                    mov_name=proteins_and_chains[2],
                    mov_chain=proteins_and_chains[3],
                    cath_level = proteins_and_chains[4],
                ))
        logging.info(f"found {len(pairs)} pairs for {ligand}")
    return pairs