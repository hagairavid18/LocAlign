import logging
from Bio.PDB import  Superimposer
from Bio.PDB.Atom import Atom
from aligners import BaseStructureAligner
import numpy as np

logger = logging.getLogger(__name__)


class SVDAligner(BaseStructureAligner):
    def __init__(self) -> None:
        super().__init__()
        self.name = "SVDaligner"
  
    def impose_structure(self, fix_points: list[Atom], src_points: list[Atom], save_dir: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        super_imposer = Superimposer()
        min_lenght = min(len(fix_points), len(src_points))
        super_imposer.set_atoms(fix_points[:min_lenght], src_points[:min_lenght])
        rot, tran = super_imposer.rotran
        return rot, tran