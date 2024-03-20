import warnings
import logging
import os
from datetime import datetime
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.Model import Model
from Bio.PDB.PDBIO import PDBIO

from objects import Protein
from alligners import BaseStructureAlligner

logger = logging.getLogger(__name__)


warnings.filterwarnings("ignore", category=PDBConstructionWarning)


class ProteinPair:
    def __init__(self, ref_proein: Protein, mov_protein: Protein, ligand_name: str, ligand_id_name: str, ref_model_idx: int = 0, mov_model_idx: int = 0) -> None:
  
        self._ref_protein: Protein = ref_proein
        self._mov_protein: Protein = mov_protein
        self._ligand_id_name = ligand_id_name
        self._ref_model_idx = ref_model_idx
        self._mov_model_idx = mov_model_idx
        
        current_time = datetime.now()
        self._base_dir = f'alligned_structures/{ligand_name}/{self._ref_protein._pdb_name}_to_{self._mov_protein._pdb_name}/{current_time.strftime("%Y-%m-%d %H:%M:%S")}'
        os.makedirs(self._base_dir, exist_ok=True)
        
        self._ref_model, self._mov_model = self._init_models()
        
    def _init_models(self, ref_model_idx: int = 0, mov_model_idx: int = 0) -> Model:
        return self._ref_protein.get_model(ref_model_idx), self._ref_protein.get_model(mov_model_idx) 
  
    def get_ref_model(self):
        pass
    
    def allign_atoms(self, alligner: BaseStructureAlligner, atom_type: str = "ligand") -> Model:
        ref_atoms = self._ref_protein.get_ligand_atoms(self._ligand_id_name)
        mov_atoms = self._mov_protein.get_ligand_atoms(self._ligand_id_name)
    
        R, t = alligner.impose_structure(ref_atoms, mov_atoms, self._base_dir)
        logging.info(f'Rotation: {R}, translation: {t}')
        copy_model = self._mov_protein.get_model(self._mov_model_idx)
        for atom in copy_model.get_atoms():
            atom.transform(R[:3, :3], t[:3])
        self.save_structre(copy_model, alligner.name)

    def save_structre(self, model: Model, alligned_by: str) -> None:        
        file_path = os.path.join(self._base_dir, f"{alligned_by}.pdb")
        io = PDBIO()
        io.set_structure(model)
        io.save(file_path)