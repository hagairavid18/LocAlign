import warnings
import logging
import os
import numpy as np
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
    def validate_ligand_pair(ligand_atoms_ref: list[Atom], ligand_atoms_mov: list[Atom], max_length_ratio: float = 1.2) -> str:
        
        if max(len(ligand_atoms_ref), len(ligand_atoms_mov)) / min(len(ligand_atoms_ref), len(ligand_atoms_mov)) > max_length_ratio:
            return f"The lengths of the lignads differ significantly. The length ratio exceeds {max_length_ratio}"
        ref_ids = set([atom.id for atom in ligand_atoms_ref]) 
        mov_ids = set([atom.id for atom in ligand_atoms_mov])
        if len(ref_ids.intersection(mov_ids)) / min(len(ligand_atoms_mov), len(ligand_atoms_ref)) < 0.8:
            return f"The lignad atoms lack sufficient overlap, with less than 80% of the smaller one having corresponding atoms in the longer one."        
        return ""

    def _apply_transformations_and_save_transformed_models(self, R: list[np.ndarray], t: list[np.ndarray], alligner: BaseStructureAlligner) -> None:
         
         for i in range(len(R)):
            copy_model = self._mov_protein.get_model(self._mov_model_idx).copy()
            
            for atom in copy_model.get_atoms():
                atom.transform(R[i][:3, :3], t[i][:3])
            
            if self._save_transformed_protein:
                self.save_structre(copy_model, alligner.name, str(i))
            only_ligand_model, _ = Protein.create_ligand_model(copy_model, self._ligand_id_name, self._mov_protein._chain_id)
            self.save_structre(only_ligand_model, alligner.name, str(i) + '_ligand')
    
    def find_transformations(self, alligner: BaseStructureAlligner, transform_ligand: bool = False, min_ligand_atoms: int = 3) -> tuple[tuple, tuple, tuple, tuple, str]:
        ref_ligand: list[list[Atom]] = self._ref_protein.get_ligand_atoms(self._ligand_id_name)
        mov_ligand: list[list[Atom]] = self._mov_protein.get_ligand_atoms(self._ligand_id_name)
        all_R, all_t, all_rmse, all_coverage = [], [], [], []
        for i, ref_residue in enumerate(ref_ligand):
            for j, mov_residue in enumerate(mov_ligand):
                if len(ref_residue) < min_ligand_atoms or len(mov_residue) < min_ligand_atoms:
                    error_message = f"One of the ligands has less than {min_ligand_atoms} atoms"
                    continue

                error_message: str = ProteinPair.validate_ligand_pair(ref_residue, mov_residue)
                if len(error_message) > 1:
                    continue
            
                R, t, rmse, coverage = alligner.impose_structure(ref_residue, mov_residue, self._base_dir)
                all_R.append(R)
                all_t.append(all_t)
                all_rmse.append(rmse)
                all_coverage.append(coverage)

                if transform_ligand:
                    self._apply_transformations_and_save_transformed_models(R, t, alligner + '_' + i + '_' + j)
        
        if len(all_R) == 0:
            return (), (), (), (), error_message
        
        return tuple(all_R), tuple(all_t), tuple(all_rmse), tuple(all_coverage), ""

    @property
    def number_of_ligand_atoms(self) -> tuple[int]:
        return self._ref_protein.get_num_of_ligand_atoms(), self._mov_protein.get_num_of_ligand_atoms()
    
    def save_structre(self, model: Model, alligned_by: str, postfix: str|None = None) -> None:        
        file_name = f"{alligned_by}.pdb" if not postfix else f"{alligned_by}_{postfix}.pdb"
        file_path = os.path.join(self._base_dir, file_name)
        io = PDBIO()
        io.set_structure(model)
        io.save(file_path)