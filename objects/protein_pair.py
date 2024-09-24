import warnings
import logging
import os
import numpy as np
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.Model import Model
from Bio.PDB.Atom import Atom
from Bio.PDB.PDBIO import PDBIO
from scipy.spatial import distance_matrix


from aligners.dali_aligner import DaliAligner
from utils.constants import LIGAND_DIR

from objects import Protein
from utils.constants import NOT_ENOUGH_ATOMS_MESSAGE, TOO_MUCH_RESIDUES_MESSAGE, LIGAND_RESIDUE_IS_MISSED_MESSAGE, N_ATOMS_RATIO_MESSAGE, LIGAND_OVERLAP_MESSAGE, ResultHolder
from aligners import BaseStructureAligner

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=PDBConstructionWarning)


def best_buddy_count(P, Q):
    dist_PQ = distance_matrix(P, Q)
    dist_QP = distance_matrix(Q, P)
    closest_in_Q_to_P = np.argmin(dist_PQ, axis=1)
    closest_in_P_to_Q = np.argmin(dist_QP, axis=1)
    
    best_buddies = 0
    for i, j in enumerate(closest_in_Q_to_P):
        if closest_in_P_to_Q[j] == i:
            best_buddies += 1
            
    return best_buddies


class ProteinPair:
    def __init__(self, ref_proein: Protein, mov_protein: Protein, ligand_name: str, ref_model_idx: int = 0,
                  mov_model_idx: int = 0, save_transformed_models: bool = False) -> None:
  
        self._ref_protein: Protein = ref_proein
        self._mov_protein: Protein = mov_protein
        self._ligand_name = ligand_name
        self._ref_model_idx = ref_model_idx
        self._mov_model_idx = mov_model_idx
        self._save_transformed_models = save_transformed_models
        
        self._base_dir = f'{LIGAND_DIR}/{ligand_name}/{self._mov_protein._pdb_name}_to_{self._ref_protein._pdb_name}'
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
                                                           aligner: BaseStructureAligner,
                                                            ref_residue_index: int = 0, mov_residue_index: int = 0) -> None:

         for i in range(len(R)):
            copy_model = self._mov_protein.get_model(self._mov_model_idx).copy()
            
            for atom in copy_model.get_atoms():
                atom.transform(R[i][:3, :3], t[i][:3])
            
            try:
                self.save_structre(copy_model, aligner.name, str(i) + '_protein_' + str(ref_residue_index) + '_' + str(mov_residue_index))
                only_ligand_model, _ = Protein.create_ligand_model(copy_model, self._ligand_name, self._mov_protein._chain_id)
                self.save_structre(only_ligand_model, aligner.name, str(i) + '_ligand_' + str(ref_residue_index) + '_' + str(mov_residue_index))
            except Exception as e:
                print(e)
    
    def _get_best_buddy_ratio(self, R, t, mov_ligand_res_idx, ref_ligand_res_idx) -> float:
        mov_atoms: np.ndarray = self._mov_protein.get_pocket_atoms(ligand_res_idx = mov_ligand_res_idx)
        ref_atoms: np.ndarray = self._ref_protein.get_pocket_atoms(ligand_res_idx = ref_ligand_res_idx)
        transformed_mov_pocket = np.dot(mov_atoms, R) + t[:3]
        bbc = best_buddy_count(transformed_mov_pocket, ref_atoms)
        logging.info(f"n bb: {bbc} bbc ratio {bbc / min(mov_atoms.shape[0], ref_atoms.shape[0])}")
        return bbc / min(mov_atoms.shape[0], ref_atoms.shape[0])
    
    def find_ligand_transformations(self, holder: ResultHolder, aligner: BaseStructureAligner, min_ligand_atoms: int = 3) -> None:
        ref_ligand: list[list[Atom]] = self._ref_protein.get_ligand_residues()
        mov_ligand: list[list[Atom]] = self._mov_protein.get_ligand_residues()
        holder.n_residues_ref_ligand = len(ref_ligand)
        holder.n_residues_mov_ligand = len(mov_ligand)
        n_ligand_pairs = len(ref_ligand) * len(mov_ligand)
        if n_ligand_pairs == 0:
            holder.failure_message =  LIGAND_RESIDUE_IS_MISSED_MESSAGE
            return
        if n_ligand_pairs > 20:
            holder.failure_message = TOO_MUCH_RESIDUES_MESSAGE
            return
        error_message = ""
        all_R, all_t, all_rmse, all_coverage, all_bbr = ([[] for _ in range(n_ligand_pairs)] for _ in range(5))
        curr_pair_idx = 0
        for i, ref_residue in enumerate(ref_ligand):
            for j, mov_residue in enumerate(mov_ligand):
                if len(ref_residue) < min_ligand_atoms or len(mov_residue) < min_ligand_atoms:
                    holder.failure_message = NOT_ENOUGH_ATOMS_MESSAGE
                    continue

                error_message: str = ProteinPair.validate_ligand_pair(ref_residue, mov_residue)
                if len(error_message) > 1:
                    continue
            
                R, t, rmse, coverage = aligner.impose_structure(ref_residue, mov_residue, self._base_dir)
                if len(R) < 1:
                    continue
                
                all_R[curr_pair_idx] = [r.tolist() for r in R]
                all_t[curr_pair_idx] = [tr.tolist() for tr in t]
                all_rmse[curr_pair_idx] = rmse
                all_coverage[curr_pair_idx] = coverage
                try:
                    all_bbr[curr_pair_idx] = [self._get_best_buddy_ratio(R[k], t[k], j , i) for k in range(len(R))]
                except Exception as e:
                    logging.info(e)
                    holder.failure_message = "Failed to compute in bbr"

                if self._save_transformed_models:
                    self._apply_transformations_and_save_transformed_models(R, t, aligner, i, j)
                curr_pair_idx +=1
        
        holder.rotations = all_R
        holder.translations = all_t
        holder.rmse = all_rmse
        holder.coverage = all_coverage
        holder.bbr = all_bbr
        holder.n_transformations = sum([len(rot) for rot in all_R])
        if len(all_R) == 0:
            holder.failure_message = error_message
    
    def find_protein_transformations(self, holder: ResultHolder, aligner: BaseStructureAligner | DaliAligner) -> tuple[tuple, tuple, tuple, tuple, str]:
        
        ref_chain = self._ref_protein.get_model(self._ref_model_idx, True)
        mov_chain = self._mov_protein.get_model(self._mov_model_idx, True)
        ref_coord, seq1 = Protein.get_residue_data(ref_chain)
        mov_coord, seq2 = Protein.get_residue_data(mov_chain)
    
        if isinstance(aligner, DaliAligner):
            R, t, rmsd, _ = aligner.impose_structure(self._ref_protein, self._mov_protein, f'{LIGAND_DIR}/{self._ligand_name}')
        else:
            R, t, rmsd, _ = aligner.impose_structure(ref_coord, mov_coord, seq1, seq2, self._base_dir)

        if self._save_transformed_models:
            self._apply_transformations_and_save_transformed_models(R, t, aligner)
        
        if len(R) > 0:
            ligand_rmsd =  self._compute_ligand_rmsd(R[0], t[0])
        else:
            ligand_rmsd = None
        
        holder.__setattr__(f"{aligner.name}_rotations", R)
        holder.__setattr__(f"{aligner.name}_translations", t)
        holder.__setattr__(f"{aligner.name}_protein_rmsd", rmsd)
        holder.__setattr__(f"{aligner.name}_rmsd", ligand_rmsd)

    def _compute_ligand_rmsd(self, R, t):
        try:
            mov_ligand_model, _ = Protein.create_ligand_model(self._mov_protein.get_model(self._mov_model_idx).copy(), self._ligand_name, self._mov_protein._chain_id)
            ref_ligand_model, _ = Protein.create_ligand_model(self._ref_protein.get_model(self._ref_model_idx).copy(), self._ligand_name, self._ref_protein._chain_id)
            for atom in mov_ligand_model.get_atoms():
                atom.transform(R[:3, :3], t[:3]) 

            mov_coors = np.vstack([atom.coord for atom in mov_ligand_model.get_atoms()])
            ref_coors = np.vstack([atom.coord for atom in ref_ligand_model.get_atoms()])

            if mov_coors.shape != ref_coors.shape:
                return None # TODO

            squared_diff = np.sum((mov_coors - ref_coors) ** 2, axis=1)
            rmsd_value = np.sqrt(np.mean(squared_diff))

            return rmsd_value
        except:
            return None
        
    @staticmethod
    def compute_rmsd(coordiantes: np.ndarray, gt_trans: np.ndarray, aligner_trans: np.ndarray) -> float:
        gt_R = gt_trans[:3,:3]
        gt_t = gt_trans[:3,3]
        transformed_points_1 = np.dot(coordiantes, gt_R) + gt_t
        aligner_R = aligner_trans[:3,:3]
        aligner_t = aligner_trans[:3,3]
        transformed_points_2 = np.dot(coordiantes, aligner_R) + aligner_t
        
        squared_diff = np.sum((transformed_points_1 - transformed_points_2) ** 2, axis=1)
        rmsd_value = np.sqrt(np.mean(squared_diff))

        return rmsd_value

    @property
    def number_of_ligand_atoms(self) -> tuple[int]:
        return self._ref_protein.get_num_of_ligand_atoms(), self._mov_protein.get_num_of_ligand_atoms()
    
    def save_structre(self, model: Model, aligned_by: str, postfix: str|None = None) -> None:        
        file_name = f"{aligned_by}.pdb" if not postfix else f"{aligned_by}_{postfix}.pdb"
        file_path = os.path.join(self._base_dir, file_name)
        io = PDBIO()
        io.set_structure(model)
        io.save(file_path)