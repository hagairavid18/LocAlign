import logging
from typing import Optional
from Bio.PDB.Atom import Atom
import numpy as np
from tmtools import tm_align

from aligners import BaseStructureAligner

logging.getLogger('matplotlib').setLevel(logging.ERROR)


logger = logging.getLogger(__name__)


class TMAligner(BaseStructureAligner):
    def __init__(self) -> None:
  
        super().__init__()
        self.name = "TMAligner"

    def impose_structure(self, fix_points: list[Atom], mov_points: list[Atom], seq1: str, seq2: str, save_dir: Optional[str] = None) -> tuple[list[np.ndarray], list[np.ndarray]]:
        res = tm_align(mov_points, fix_points, seq2, seq1)
        return [np.linalg.inv(res.u)], [res.t], res.tm_norm_chain2, None