import logging
import subprocess
import numpy as np
import os

logger = logging.getLogger(__name__)


def kabsch_from_correspondence_matrix_numpy(P: np.ndarray, Q: np.ndarray, C: np.ndarray):
    """
    Computes rotation R and translation t to align P -> Q.

    Aligned coordinates are given by:
        aligned = np.dot(P, R) + t

    Args:
        P: [K1, 3] source points
        Q: [K2, 3] target points
        C: [K1, K2] binary correspondence matrix (C[i,j] = 1 if P[i] -> Q[j])

    Returns:
        R: [3, 3] rotation matrix
        t: [3] translation vector
        rmsd: scalar RMSD value
    """
    if C.shape != (P.shape[0], Q.shape[0]):
        logger.error(f"Shape mismatch: C {C.shape}, P {P.shape}, Q {Q.shape}")
        # print(f"Shape mismatch: C {C.shape}, P {P.shape}, Q {Q.shape}")
        raise ValueError("C must be [K1, K2]")

    # Get matched pairs
    idx_P, idx_Q = np.nonzero(C)
    matched_P = P[idx_P]
    matched_Q = Q[idx_Q]

    if matched_P.shape[0] == 0:
        raise ValueError("No correspondences found in C.")

    # Compute centroids
    centroid_P = matched_P.mean(axis=0)
    centroid_Q = matched_Q.mean(axis=0)

    # Center
    P_centered = matched_P - centroid_P
    Q_centered = matched_Q - centroid_Q

    # Covariance
    H = Q_centered.T @ P_centered

    # SVD
    U, _, Vt = np.linalg.svd(H)
    det_sign = np.sign(np.linalg.det(Vt.T @ U.T))
    eye = np.eye(3)
    eye[-1, -1] = det_sign
    R = Vt.T @ eye @ U.T

    # Translation
    t = centroid_Q - centroid_P @ R

    # RMSD
    aligned_P = matched_P @ R + t
    rmsd = np.sqrt(np.mean(np.sum((aligned_P - matched_Q) ** 2, axis=1)))

    return R, t, rmsd



class SoftAlignAligner():
    HOME_PATH = "/home/iscb/wolfson/hagairavid"
    def __init__(self) -> None:  
        self.name = "SoftAlignAligner"

    def impose_structure(self, ref_protein, mov_protein, ref_coord, mov_coord, ligand_dir: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
       
        mov_name, mov_chain = mov_protein._pdb_name, mov_protein._chain_id
        ref_name, ref_chain = ref_protein._pdb_name, ref_protein._chain_id
        mov_path = os.path.join(ligand_dir, mov_name + mov_chain + '_non_ligand_.ent')
        ref_path = os.path.join(ligand_dir, ref_name + ref_chain + '_non_ligand_.ent')
        try:
            align_log = subprocess.run(["/home/iscb/wolfson/hagairavid/miniforge3/envs/miner/bin/python", "/home/iscb/wolfson/hagairavid/SoftAlign/align.py", "--pdb_source", "custom", "--pdb1_id", ref_path, "--pdb2_id", mov_path, "--output_dir", ligand_dir, "--model_path", "/home/iscb/wolfson/hagairavid/SoftAlign/models/CONT_SW_05_T_3_1"],
                                       capture_output=True, text=True)
            npy_path = os.path.join(ligand_dir, f"soft_alignment_{ref_name}{ref_chain}_vs_{mov_name}{mov_chain}.npy")
            if not os.path.exists(npy_path):
                logger.error(f"SoftAlign output file {npy_path} not found. Ensure the alignment was successful.")
                return [], [], [], []
            with open(npy_path, 'rb') as f:
                matrices = np.load(f, allow_pickle=True)
            R, t, rmsd = kabsch_from_correspondence_matrix_numpy(mov_coord, ref_coord, matrices.T)    # transpose!
            if align_log.returncode != 0:
                logger.log(f"Error in SoftAlign alignment: {align_log.stderr}")
                return [], [], [], []
  
       
            return [np.array(R)], [np.array(t)], float(rmsd), []

        except Exception as e:
            logger.error(f"Error during SoftAlign alignment: {e}")
            print(f"Error during SoftAlign alignment: {e}")
            return [], [], [], []