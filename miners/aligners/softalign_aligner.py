import logging
import subprocess
import numpy as np
import os

logger = logging.getLogger(__name__)





class SoftAlignAligner():
    HOME_PATH = "/home/iscb/wolfson/hagairavid"
    def __init__(self) -> None:  
        self.name = "SoftAlignAligner"

    def impose_structure(self, tar_protein, src_protein, ligand_dir: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
       
        src_name, src_chain = src_protein._pdb_name, src_protein._chain_id
        tar_name, tar_chain = tar_protein._pdb_name, tar_protein._chain_id
        src_path = os.path.join(ligand_dir, src_name + src_chain + '_non_ligand_.ent')
        tar_path = os.path.join(ligand_dir, tar_name + tar_chain + '_non_ligand_.ent')
        try:
            align_log = subprocess.run(["/home/iscb/wolfson/hagairavid/miniforge3/envs/miner/bin/python", "/home/iscb/wolfson/hagairavid/SoftAlign/align.py", "--pdb_source", "custom", "--pdb1_id", tar_path, "--pdb2_id", src_path, "--output_dir", ligand_dir, "--model_path", "/home/iscb/wolfson/hagairavid/SoftAlign/models/CONT_SW_05_T_3_1"],
            # align_log = subprocess.run(["/home/iscb/wolfson/hagairavid/miniforge3/envs/miner/bin/python", "/home/iscb/wolfson/hagairavid/SoftAlign/align.py", "--pdb_source", "custom", "--pdb1_id", tar_path, "--pdb2_id", src_path, "--output_dir", ligand_dir, "--model_type", "Smith-Waterman"],
                                       capture_output=True, text=True)
            npy_path = os.path.join(ligand_dir, f"soft_alignment_{tar_name}{tar_chain}_vs_{src_name}{src_chain}.npz")
            if not os.path.exists(npy_path):
                logger.error(f"SoftAlign output file {npy_path} not found. Ensure the alignment was successful.")
                return [], [], [], []
            with open(npy_path, 'rb') as f:
                matrices = np.load(f, allow_pickle=True)
                R = matrices["R"]
                t = matrices["t"]
                corr_rmsd = matrices["rmsd"]
            # R, t, rmsd = kabsch_from_correspondence_matrix_numpy(src_coord, tar_coord, matrices.T)    # transpose!
            if align_log.returncode != 0:
                logger.log(f"Error in SoftAlign alignment: {align_log.stderr}")
                return [], [], [], []
  
       
            return [np.array(R)], [np.array(t)], float(corr_rmsd), []

        except Exception as e:
            logger.error(f"Error during SoftAlign alignment: {e}")
            print(f"Error during SoftAlign alignment: {e}")
            return [], [], [], []