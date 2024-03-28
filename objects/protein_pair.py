import warnings
import logging
import os
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.Model import Model
from Bio.PDB.Atom import Atom
from Bio.PDB.PDBIO import PDBIO

from objects import Protein
from alligners import BaseStructureAlligner

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=PDBConstructionWarning)


class ProteinPair:
    def __init__(self, ref_proein: Protein, mov_protein: Protein, ligand_name: str, ligand_id_name: str,
                  ref_model_idx: int = 0, mov_model_idx: int = 0, save_transformed_protein: bool = False) -> None:
  
        self._ref_protein: Protein = ref_proein
        self._mov_protein: Protein = mov_protein
        self._ligand_id_name = ligand_id_name
        self._ref_model_idx = ref_model_idx
        self._mov_model_idx = mov_model_idx
        self._save_transformed_protein = save_transformed_protein
        
        self._base_dir = f'alligned_structures/{ligand_name}/{self._mov_protein._pdb_name}_to_{self._ref_protein._pdb_name}'
        os.makedirs(self._base_dir, exist_ok=True)
        
        self._ref_model, self._mov_model = self._init_models()
        
    def _init_models(self, ref_model_idx: int = 0, mov_model_idx: int = 0) -> tuple[Model, Model]:
        return self._ref_protein.get_model(ref_model_idx), self._ref_protein.get_model(mov_model_idx) 
    
    @staticmethod
    def validate_ligand_pair(ligand_atoms_ref: list[Atom], ligand_atoms_mov: list[Atom]) -> bool:
        
        if len(ligand_atoms_ref) != len(ligand_atoms_mov):
            logger.debug(f"ref ligand has {len(ligand_atoms_ref)} and mov ligand has {len(ligand_atoms_mov)}")
            return False

        for atom_idx in range(len(ligand_atoms_ref)):
            if ligand_atoms_ref[atom_idx].id != ligand_atoms_mov[atom_idx].id:
                logger.debug(f"Ligand atoms are different between he two proteins")
                return False
        return True

    def allign_atoms(self, alligner: BaseStructureAlligner, atom_type: str = "ligand") -> int:
        ref_ligand: list[Atom] = self._ref_protein.get_ligand_atoms(self._ligand_id_name)
        mov_ligand: list[Atom] = self._mov_protein.get_ligand_atoms(self._ligand_id_name)

        if not ProteinPair.validate_ligand_pair(ref_ligand, mov_ligand):
            return 0
     
        R, t = alligner.impose_structure(ref_ligand, mov_ligand, self._base_dir)
        if not isinstance(R, list):
            R = [R]
            t = [t]
        
        for i in range(len(R)):
            copy_model = self._mov_protein.get_model(self._mov_model_idx).copy()
            
            for atom in copy_model.get_atoms():
                atom.transform(R[i][:3, :3], t[i][:3])
            
            if self._save_transformed_protein:
                self.save_structre(copy_model, alligner.name, str(i))
            only_ligand_model, _ = Protein.create_ligand_model(copy_model, self._ligand_id_name, self._mov_protein._chain_id)
            self.save_structre(only_ligand_model, alligner.name, str(i) + '_ligand')
        return len(R)

    def save_structre(self, model: Model, alligned_by: str, postfix: str|None = None) -> None:        
        file_name = f"{alligned_by}.pdb" if not postfix else f"{alligned_by}_{postfix}.pdb"
        file_path = os.path.join(self._base_dir, file_name)
        io = PDBIO()
        io.set_structure(model)
        io.save(file_path)