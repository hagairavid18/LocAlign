import logging
import re
import shutil
import subprocess
import tempfile
# from objects import Protein 
import numpy as np
import os


logging.getLogger('matplotlib').setLevel(logging.ERROR)


logger = logging.getLogger(__name__)


class DaliAligner():
    HOME_PATH = "/home/iscb/wolfson/hagairavid"
    DAT_PATH = os.path.join(HOME_PATH, "DaliLite.v5/DAT")
    IMPORT_PATH = os.path.join(HOME_PATH, "DaliLite.v5/bin/import.pl")
    DALI_PATH = os.path.join(HOME_PATH, "DaliLite.v5/bin/dali.pl")
    def __init__(self) -> None:  
        self.name = "DaliAligner"
        self._check_paths()
    
    def _check_paths(self):
        assert os.path.exists(self.HOME_PATH), f"{self.HOME_PATH} dosn't exist, can't use Dali"
        assert os.path.exists(self.DAT_PATH), f"{self.DAT_PATH} dosn't exist, can't use Dali"
        assert os.path.exists(self.IMPORT_PATH), f"{self.IMPORT_PATH} dosn't exist, can't use Dali"
        assert os.path.exists(self.DALI_PATH), f"{self.DALI_PATH} dosn't exist, can't use Dali"

    @staticmethod
    def extract_matrices_combined(file_path: str) -> tuple[np.ndarray, float, float]:
        matrices = []
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                if i == 3:
                    measure_line = line
                if line.startswith("-matrix"):
                    match = re.findall(r'-?\d+\.\d+', line)
                    if match:
                        matrices.append([float(num) for num in match])
            
            if len(matrices) == 3:
                file.seek(0)
                rmsd = measure_line.split('  ')[3]
                z = measure_line.split(' ')[6]
                return np.array(matrices), rmsd, z
        return None, None, None

    def impose_structure(self, tar_protein, src_protein, ligand_dir: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
       
        src_name, src_chain = src_protein._pdb_name, src_protein._chain_id
        tar_name, tar_chain = tar_protein._pdb_name, tar_protein._chain_id
        src_path = os.path.join('..', src_name + src_chain + '_non_ligand_.ent')
        tar_path = os.path.join('..', tar_name + tar_chain + '_non_ligand_.ent')
        try:
            temp_dir = tempfile.mkdtemp(ptarix=os.path.join(self.HOME_PATH, "ligand_aligner", ligand_dir + '/'))
            os.makedirs(temp_dir, exist_ok=True)
            os.chdir(temp_dir)
            import_1 = subprocess.run([self.IMPORT_PATH, '--pdbfile', src_path, '--pdbid', src_name, '--dat', self.DAT_PATH], capture_output=True, text=True, check=True)
            import_2 = subprocess.run([self.IMPORT_PATH, '--pdbfile', tar_path, '--pdbid', tar_name, '--dat', self.DAT_PATH], capture_output=True, text=True, check=True)
            align_log = subprocess.run([self.DALI_PATH, '--cd1', tar_name + tar_chain , '--cd2', src_name + src_chain,
                                          '--dat1', self.DAT_PATH, '--dat2', self.DAT_PATH, '--title',
                                            "output options" ,'--outfmt', "summary,alignments,equivalences,transrot", "--clean"
                                              ], capture_output=True, text=True, check=True)

            matrix, rmsd, _ = DaliAligner.extract_matrices_combined(f'{tar_name}{tar_chain}.txt')
            os.chdir('/home/iscb/wolfson/hagairavid/LocAlign')
            try:
                shutil.rmtree(temp_dir)
            except OSError as e:
                logging.info(e)
                pass
  
            if matrix is not None:
                R = np.linalg.inv(matrix[:, :3])
                t = matrix[:, 3]
                return [R], [t], float(rmsd), []
            else:
                return [], [], [], []
            
        except Exception as e:
            os.chdir('/home/iscb/wolfson/hagairavid/ligand_aligner')
            logging.info(e)
            try:
                shutil.rmtree(temp_dir)
            except:
                logging.info(f'didnt remove dir {temp_dir}')
                pass
            return [], [], [], []