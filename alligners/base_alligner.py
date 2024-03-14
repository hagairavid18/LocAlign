import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import numpy as np
from Bio.PDB.Atom import Atom

warnings.filterwarnings("ignore", category=PDBConstructionWarning)



class BaseStructureAlligner:
    def __init__(self) -> None:
        self.name: str = "base"
        pass
    
    def impose_structure(self, fix_points: list[Atom], mov_points: list[Atom]) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Subclass must implement abstract method")



  





