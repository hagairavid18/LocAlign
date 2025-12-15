import warnings
import logging
import os
import numpy as np
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.Model import Model
from Bio.PDB.Atom import Atom
from Bio.PDB.PDBIO import PDBIO
from scipy.spatial import distance_matrix


from miners.utils.constants import LIGAND_DIR, BaselineHolder

from miners.objects import Protein
from miners.utils.constants import  TOO_MUCH_RESIDUES_MESSAGE, LIGAND_RESIDUE_IS_MISSED_MESSAGE, N_ATOMS_RATIO_MESSAGE, LIGAND_OVERLAP_MESSAGE, ResultHolder

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=PDBConstructionWarning)


from scipy.spatial import distance_matrix
import numpy as np

from scipy.spatial.transform import Rotation
import numpy as np


def best_buddy_count(P, Q, dist_thresh=None):
    """
    Count the number of best buddies between two point sets P and Q.
    
    Args:
        P (np.ndarray): Point set P (N x D array).
        Q (np.ndarray): Point set Q (M x D array).
        dist_thresh (float, optional): Distance threshold. If provided, only pairs
                                       within this distance are counted as best buddies.

    Returns:
        int: Number of best buddies.
    """
    dist_PQ = distance_matrix(P, Q)
    dist_QP = distance_matrix(Q, P)
    closest_in_Q_to_P = np.argmin(dist_PQ, axis=1)
    closest_in_P_to_Q = np.argmin(dist_QP, axis=1)
    best_buddies = 0
    for i, j in enumerate(closest_in_Q_to_P):
        if closest_in_P_to_Q[j] == i:
            if dist_thresh is None or dist_PQ[i, j] < dist_thresh:
                best_buddies += 1

    return best_buddies



class ProteinPair:
    def __init__(self, tar_proein: Protein, src_protein: Protein, ligand_name: str, tar_model_idx: int = 0,
                  src_model_idx: int = 0, save_transformed_models: bool = False, ligand_dir: str = LIGAND_DIR) -> None:
  
        self._tar_protein: Protein = tar_proein
        self._src_protein: Protein = src_protein
        self._ligand_name = ligand_name
        self._ligand_dir = ligand_dir
        self._tar_model_idx = tar_model_idx
        self._src_model_idx = src_model_idx
        self._save_transformed_models = save_transformed_models
        
        # self._base_dir = f'{ligand_dir}/{ligand_name}/{self._src_protein._pdb_name}{self._src_protein._chain_id}_to_{self._tar_protein._pdb_name}{self._tar_protein._chain_id}'
        # os.makedirs(self._base_dir, exist_ok=True)
        
        self._tar_model, self._src_model = self._init_models()
        
    def _init_models(self, tar_model_idx: int = 0, src_model_idx: int = 0) -> tuple[Model, Model]:
        return self._tar_protein.get_model(tar_model_idx), self._tar_protein.get_model(src_model_idx) 
    
    @staticmethod
    def validate_ligand_pair(ligand_atoms_tar: list[Atom], ligand_atoms_src: list[Atom], max_length_ratio: float = 1.2) -> str:
        
        if max(len(ligand_atoms_tar), len(ligand_atoms_src)) / min(len(ligand_atoms_tar), len(ligand_atoms_src)) > max_length_ratio:
            return N_ATOMS_RATIO_MESSAGE
        tar_ids = set([atom.id for atom in ligand_atoms_tar]) 
        src_ids = set([atom.id for atom in ligand_atoms_src])
        if len(tar_ids.intersection(src_ids)) / min(len(ligand_atoms_src), len(ligand_atoms_tar)) < 0.8:
            return LIGAND_OVERLAP_MESSAGE        
        return ""

    def _apply_transformations_and_save_transformed_models(self, R: list[np.ndarray], t: list[np.ndarray],
                                                           aligner,
                                                            tar_residue_index: int = 0, src_residue_index: int = 0) -> None:

         for i in range(len(R)):
            copy_model = self._src_protein.get_model(self._src_model_idx).copy()
            
            for atom in copy_model.get_atoms():
                atom.transform(R[i][:3, :3], t[i][:3])
            
            try:
                self.save_structre(copy_model, aligner.name, str(i) + '_protein_' + str(tar_residue_index) + '_' + str(src_residue_index))
                only_ligand_model, _ = Protein.create_ligand_model(copy_model, self._ligand_name, self._src_protein._chain_id)
                self.save_structre(only_ligand_model, aligner.name, str(i) + '_ligand_' + str(tar_residue_index) + '_' + str(src_residue_index))
            except Exception as e:
                print(e)
    
    def _get_best_buddy_ratio(self, R, t, src_ligand_res_idx, tar_ligand_res_idx, bb_thresh: None | float = None) -> float:
        src_atoms, _ = self._src_protein.get_pocket_atoms_within_4A(ligand_res_idx = src_ligand_res_idx, distance_thresh=4.0)
        tar_atoms, _ = self._tar_protein.get_pocket_atoms_within_4A(ligand_res_idx = tar_ligand_res_idx, distance_thresh=4.0)
        transformed_src_pocket = np.dot(src_atoms, R) + t[:3]
        bbc = best_buddy_count(transformed_src_pocket, tar_atoms, bb_thresh)
        bb_ratio = bbc / min(src_atoms.shape[0], tar_atoms.shape[0])
        logging.info(f"4 ang n bb: {bbc} bbc ratio {bb_ratio}")
        return bb_ratio, bbc
    
    def _get_best_buddy_around_tar_center(self, R, t, bb_thresh: None | float = None, distance_thresh: float = 4.0) -> float:
        
        all_atoms_tar = np.array([atom.coord for atom in self._tar_protein._get_atoms(all_atoms=True) if atom.element != "H"])
        tar_close_atoms = Protein.get_atoms_within_distance(all_atoms_tar ,center=all_atoms_tar.mean(0), distance_thresh=distance_thresh)
        
        copy_model = self._src_protein.get_model(self._src_model_idx).copy()
        for atom in copy_model.get_atoms():
            atom.transform(R[:3, :3], t[:3])
        transformed_src_coord = [atom.coord for atom in copy_model.get_atoms() if atom.element != "H"]
        
        src_close_atoms = Protein.get_atoms_within_distance(transformed_src_coord, center=all_atoms_tar.mean(0), distance_thresh=distance_thresh)
        
        # Compute best buddy count and ratio
        bbc = best_buddy_count(src_close_atoms, tar_close_atoms, bb_thresh)
        bbr = bbc / min(src_close_atoms.shape[0], tar_close_atoms.shape[0])
        logging.info(f"{distance_thresh}A n bb: {bbc} bb ratio {bbr}")
        
        return bbr, bbc
    
    def find_ligand_transformations(self, holder: ResultHolder, aligner, min_ligand_atoms: int = 10) -> None:
        tar_ligand: list[list[Atom]] = self._tar_protein.get_ligand_residues()
        src_ligand: list[list[Atom]] = self._src_protein.get_ligand_residues()
        holder.n_residues_tar_ligand = len(tar_ligand)
        holder.n_residues_src_ligand = len(src_ligand)
        n_ligand_pairs = len(tar_ligand) * len(src_ligand)
        if n_ligand_pairs == 0:
            holder.failure_message =  LIGAND_RESIDUE_IS_MISSED_MESSAGE
            return
        if n_ligand_pairs > 1:
            holder.failure_message = TOO_MUCH_RESIDUES_MESSAGE
            return
        error_message = ""
        all_R, all_t, all_rmse, all_coverage, all_bbr, all_bbc = ([[] for _ in range(n_ligand_pairs)] for _ in range(6))
        curr_pair_idx = 0
        for i, tar_residue in enumerate(tar_ligand):
            for j, src_residue in enumerate(src_ligand):
                # if len(tar_residue) < min_ligand_atoms or len(src_residue) < min_ligand_atoms:
                #     holder.failure_message = NOT_ENOUGH_ATOMS_MESSAGE
                #     continue

                error_message: str = ProteinPair.validate_ligand_pair(tar_residue, src_residue)
                if len(error_message) > 1:
                    continue
            
                # R, t, rmse, coverage = aligner.impose_structure(tar_residue, src_residue, self._base_dir)
                # if len(R) < 1:
                #     continue
                
                # all_R[curr_pair_idx] = [r.tolist() for r in R]
                # all_t[curr_pair_idx] = [tr.tolist() for tr in t]
                # all_rmse[curr_pair_idx] = rmse
                # all_coverage[curr_pair_idx] = coverage
                # try:
                #     for k in range(len(R)):
                #         bbr, bbc = self._get_best_buddy_ratio(R[k], t[k], j , i, bb_thresh=2.0)
                #         all_bbr[curr_pair_idx].append(bbr)
                #         all_bbc[curr_pair_idx].append(bbc)
                # except Exception as e:
                #     logging.info(e)
                #     holder.failure_message = "Failed to compute in bbr"

                # if self._save_transformed_models:
                #     if len(R) > 1:
                #         print(f"saving {len(R)} models for ligand {self._ligand_name} for pair {self._tar_protein._pdb_name}{self._tar_protein._chain_id} to {self._src_protein._pdb_name}{self._src_protein._chain_id}")
                #     self._apply_transformations_and_save_transformed_models(R, t, aligner, i, j)
                curr_pair_idx +=1
        
        holder.rotations = all_R
        holder.translations = all_t
        holder.rmse = all_rmse
        holder.coverage = all_coverage
        holder.bbr = all_bbr
        holder.bbc = all_bbc
        holder.n_transformations = sum([len(rot) for rot in all_R])
        if len(all_R) == 0:
            holder.failure_message = error_message
    
    def find_protein_transformations(self, holder: BaselineHolder, aligner) -> None:
        
        tar_chain = self._tar_protein.get_model(self._tar_model_idx, True)
        src_chain = self._src_protein.get_model(self._src_model_idx, True)
        tar_coord, seq1, _ = Protein.get_residue_data(tar_chain)
        src_coord, seq2, _ = Protein.get_residue_data(src_chain)
        if aligner.name in  ["DaliAligner", "SoftAlignAligner"]:
            R, t, rmsd, _ = aligner.impose_structure(self._tar_protein, self._src_protein, f'{self._ligand_dir}/{self._ligand_name}')
        else:
            R, t, rmsd, _ = aligner.impose_structure(tar_coord, src_coord, seq1, seq2)

        if self._save_transformed_models:
            self._apply_transformations_and_save_transformed_models(R, t, aligner)
        
        if len(R) > 0:
            ligand_rmsd =  self._compute_ligand_rmsd(R[0], t[0])
        #     try:

        #         tar_ligand: list[list[Atom]] = self._tar_protein.get_ligand_residues()
        #         src_ligand: list[list[Atom]] = self._src_protein.get_ligand_residues()
        #         n_ligand_pairs = len(tar_ligand) * len(src_ligand)
        #         all_bbr, all_bbc = ([[] for _ in range(n_ligand_pairs)] for _ in range(2))
        #         curr_pair_idx = 0
        #         for i, tar_residue in enumerate(tar_ligand):
        #             for j, src_residue in enumerate(src_ligand):
        #                     error_message: str = ProteinPair.validate_ligand_pair(tar_residue, src_residue)
        #                     if len(error_message) > 1:
        #                         continue
        #                     for k in range(len(R)):
        #                         bbr, bbc = self._get_best_buddy_ratio(R[k], t[k],j, i, bb_thresh=2.0)
        #                         all_bbr[curr_pair_idx].append(bbr)
        #                         all_bbc[curr_pair_idx].append(bbc)
        #                     curr_pair_idx +=1
        #     except Exception as e:
        #         logging.info(e)
        #         holder.failure_message = "Failed to compute in bbr"
        else:
            ligand_rmsd = None
        
        holder.__setattr__(f"{aligner.name}_rotations", R)
        holder.__setattr__(f"{aligner.name}_translations", t)
        holder.__setattr__(f"{aligner.name}_ligand_rmsd", ligand_rmsd)
        # holder.__setattr__(f"{aligner.name}_bbr", all_bbr)
        # holder.__setattr__(f"{aligner.name}_bbc", all_bbc)

    def _compute_ligand_rmsd(self, R, t):
        try:
            src_ligand_model, _ = Protein.create_ligand_model(self._src_protein.get_model(self._src_model_idx).copy(), self._ligand_name, self._src_protein._chain_id)
            tar_ligand_model, _ = Protein.create_ligand_model(self._tar_protein.get_model(self._tar_model_idx).copy(), self._ligand_name, self._tar_protein._chain_id)
            for atom in src_ligand_model.get_atoms():
                atom.transform(R[:3, :3], t[:3]) 

            src_coors = np.vstack([atom.coord for atom in src_ligand_model.get_atoms()])
            tar_coors = np.vstack([atom.coord for atom in tar_ligand_model.get_atoms()])

            if src_coors.shape != tar_coors.shape:
                return None # TODO

            squared_diff = np.sum((src_coors - tar_coors) ** 2, axis=1)
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
        return self._tar_protein.get_num_of_ligand_atoms(), self._src_protein.get_num_of_ligand_atoms()
    
    def save_structre(self, model: Model, aligned_by: str, postfix: str|None = None) -> None:        
        file_name = f"{aligned_by}.pdb" if not postfix else f"{aligned_by}_{postfix}.pdb"
        file_path = os.path.join(self._base_dir, file_name)
        io = PDBIO()
        io.set_structure(model)
        io.save(file_path)