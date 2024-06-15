import warnings
import logging
import os
import numpy as np
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.Model import Model
from Bio.PDB.Atom import Atom
from Bio.PDB.PDBIO import PDBIO

from alligners.dali_alligner import DaliAligner
from objects import Protein
from utils.constants import NOT_ENOUGH_ATOMS_MESSAGE, TOO_MUCH_RESIDUES_MESSAGE, LIGAND_RESIDUE_IS_MISSED_MESSAGE, N_ATOMS_RATIO_MESSAGE, LIGAND_OVERLAP_MESSAGE, ResultHolder
from alligners import BaseStructureAlligner

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=PDBConstructionWarning)


class ProteinPair:
    def __init__(self, ref_proein: Protein, mov_protein: Protein, ligand_name: str, ref_model_idx: int = 0,
                  mov_model_idx: int = 0, save_transformed_protein: bool = False) -> None:
  
        self._ref_protein: Protein = ref_proein
        self._mov_protein: Protein = mov_protein
        self._ligand_name = ligand_name
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
            return N_ATOMS_RATIO_MESSAGE
        ref_ids = set([atom.id for atom in ligand_atoms_ref]) 
        mov_ids = set([atom.id for atom in ligand_atoms_mov])
        if len(ref_ids.intersection(mov_ids)) / min(len(ligand_atoms_mov), len(ligand_atoms_ref)) < 0.8:
            return LIGAND_OVERLAP_MESSAGE        
        return ""

    def _apply_transformations_and_save_transformed_models(self, R: list[np.ndarray], t: list[np.ndarray],
                                                           alligner: BaseStructureAlligner,
                                                            ref_residue_index: int = 0, mov_residue_index: int = 0) -> None:
         
         for i in range(len(R)):
            copy_model = self._mov_protein.get_model(self._mov_model_idx).copy()
            
            for atom in copy_model.get_atoms():
                atom.transform(R[i][:3, :3], t[i][:3])
            
            if self._save_transformed_protein:
                self.save_structre(copy_model, alligner.name, str(i) + '_protein_' + str(ref_residue_index) + '_' + str(mov_residue_index))
            only_ligand_model, _ = Protein.create_ligand_model(copy_model, self._ligand_name, self._mov_protein._chain_id)
            self.save_structre(only_ligand_model, alligner.name, str(i) + '_ligand_' + str(ref_residue_index) + '_' + str(mov_residue_index))
    
    def find_ligand_transformations(self, holder: ResultHolder, alligner: BaseStructureAlligner,
                                     transform_protein: bool = False, min_ligand_atoms: int = 3) -> None:
        ref_ligand: list[list[Atom]] = self._ref_protein.get_ligand_residues()
        mov_ligand: list[list[Atom]] = self._mov_protein.get_ligand_residues()
        all_R, all_t, all_rmse, all_coverage = [], [], [], []
        holder.n_residues_ref_ligand = len(ref_ligand)
        holder.n_residues_mov_ligand = len(mov_ligand)
        if len(ref_ligand) == 0 or len(mov_ligand) == 0:
            holder.failure_message =  LIGAND_RESIDUE_IS_MISSED_MESSAGE
            return
        
        if len(ref_ligand) * len(mov_ligand) > 20:
            holder.failure_message = TOO_MUCH_RESIDUES_MESSAGE
            return
        error_message = ""
        for i, ref_residue in enumerate(ref_ligand):
            for j, mov_residue in enumerate(mov_ligand):
                if len(ref_residue) < min_ligand_atoms or len(mov_residue) < min_ligand_atoms:
                    holder.failure_message = NOT_ENOUGH_ATOMS_MESSAGE
                    all_R.append([])
                    all_t.append([])
                    all_rmse.append(-1)
                    all_coverage.append(-1)
                    continue

                error_message: str = ProteinPair.validate_ligand_pair(ref_residue, mov_residue)
                if len(error_message) > 1:
                    all_R.append([])
                    all_t.append([])
                    all_rmse.append(-1)
                    all_coverage.append(-1)
                    continue
            
                R, t, rmse, coverage = alligner.impose_structure(ref_residue, mov_residue, self._base_dir)
                all_R.append([r.tolist() for r in R])
                all_t.append([tr.tolist() for tr in t])
                all_rmse.append(rmse)
                all_coverage.append(coverage)

                if transform_protein:
                    self._apply_transformations_and_save_transformed_models(R, t, alligner, i, j)
        

        if len(all_R) == 3:
            print('here')
        holder.rotations = all_R
        holder.translations = all_t
        holder.rmse = all_rmse
        holder.coverage = all_coverage
        holder.n_transformations = sum([len(rot) for rot in all_R])
        if len(all_R) == 0:
            holder.failure_message = error_message
    
    def find_protein_transformations(self, holder: ResultHolder, alligner: BaseStructureAlligner | DaliAligner,
                                      transform_protein: bool = False) -> tuple[tuple, tuple, tuple, tuple, str]:
        
        ref_chain = self._ref_protein.get_model(self._ref_model_idx, True)
        mov_chain = self._mov_protein.get_model(self._mov_model_idx, True)
        ref_coord, seq1 = Protein.get_residue_data(ref_chain)
        mov_coord, seq2 = Protein.get_residue_data(mov_chain)
    
        if isinstance(alligner, DaliAligner):
            R, t, rmsd, _ = alligner.impose_structure(self._ref_protein, self._mov_protein, f'alligned_structures/{self._ligand_name}')
        else:
            R, t, rmsd, _ = alligner.impose_structure(ref_coord, mov_coord, seq1, seq2, self._base_dir)

        if transform_protein:
            self._apply_transformations_and_save_transformed_models(R, t, alligner)
        
        holder.__setattr__(f"{alligner.name}_rotations", R)
        holder.__setattr__(f"{alligner.name}_translations", t)
        holder.__setattr__(f"{alligner.name}_rmsd", rmsd)
        # holder.p_translations = t
        # holder.p_rmsd = rmsd # TODO: handle rmsd

    @staticmethod
    def compute_rmsd(coordiantes: list[Atom], gt_trans: np.ndarray, aligner_trans: np.ndarray) -> float:
        points_homogeneous = np.hstack([coordiantes, np.ones((coordiantes.shape[0], 1))])
        transformed_points_1 = (gt_trans @ points_homogeneous.T).T[:, :3]
        transformed_points_2 = (aligner_trans @ points_homogeneous.T).T[:, :3]
        
        squared_diff = np.sum((transformed_points_1 - transformed_points_2) ** 2, axis=1)
        rmsd_value = np.sqrt(np.mean(squared_diff))

        return rmsd_value

    @property
    def number_of_ligand_atoms(self) -> tuple[int]:
        return self._ref_protein.get_num_of_ligand_atoms(), self._mov_protein.get_num_of_ligand_atoms()
    
    def save_structre(self, model: Model, alligned_by: str, postfix: str|None = None) -> None:        
        file_name = f"{alligned_by}.pdb" if not postfix else f"{alligned_by}_{postfix}.pdb"
        file_path = os.path.join(self._base_dir, file_name)
        io = PDBIO()
        io.set_structure(model)
        io.save(file_path)