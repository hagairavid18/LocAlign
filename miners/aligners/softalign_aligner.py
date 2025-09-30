import logging
import subprocess
import numpy as np
import os

logger = logging.getLogger(__name__)





class SoftAlignAligner():
    HOME_PATH = "/home/iscb/wolfson/hagairavid"
    def __init__(self) -> None:  
        self.name = "SoftAlignAligner"

    def impose_structure(self, ref_protein, mov_protein, ligand_dir: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
       
        mov_name, mov_chain = mov_protein._pdb_name, mov_protein._chain_id
        ref_name, ref_chain = ref_protein._pdb_name, ref_protein._chain_id
        mov_path = os.path.join(ligand_dir, mov_name + mov_chain + '_non_ligand_.ent')
        ref_path = os.path.join(ligand_dir, ref_name + ref_chain + '_non_ligand_.ent')
        try:
            align_log = subprocess.run(["/home/iscb/wolfson/hagairavid/miniforge3/envs/miner/bin/python", "/home/iscb/wolfson/hagairavid/SoftAlign/align.py", "--pdb_source", "custom", "--pdb1_id", ref_path, "--pdb2_id", mov_path, "--output_dir", ligand_dir, "--model_path", "/home/iscb/wolfson/hagairavid/SoftAlign/models/CONT_SW_05_T_3_1"],
            # align_log = subprocess.run(["/home/iscb/wolfson/hagairavid/miniforge3/envs/miner/bin/python", "/home/iscb/wolfson/hagairavid/SoftAlign/align.py", "--pdb_source", "custom", "--pdb1_id", ref_path, "--pdb2_id", mov_path, "--output_dir", ligand_dir, "--model_type", "Smith-Waterman"],
                                       capture_output=True, text=True)
            npy_path = os.path.join(ligand_dir, f"soft_alignment_{ref_name}{ref_chain}_vs_{mov_name}{mov_chain}.npz")
            if not os.path.exists(npy_path):
                logger.error(f"SoftAlign output file {npy_path} not found. Ensure the alignment was successful.")
                return [], [], [], []
            with open(npy_path, 'rb') as f:
                matrices = np.load(f, allow_pickle=True)
                R = matrices["R"]
                t = matrices["t"]
                rmsd = matrices["rmsd"]
            # R, t, rmsd = kabsch_from_correspondence_matrix_numpy(mov_coord, ref_coord, matrices.T)    # transpose!
            if align_log.returncode != 0:
                logger.log(f"Error in SoftAlign alignment: {align_log.stderr}")
                return [], [], [], []
  
       
            return [np.array(R)], [np.array(t)], float(rmsd), []

        except Exception as e:
            logger.error(f"Error during SoftAlign alignment: {e}")
            print(f"Error during SoftAlign alignment: {e}")
            return [], [], [], []