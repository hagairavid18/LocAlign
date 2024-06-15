import logging
import re
import subprocess
# from objects import Protein 
import numpy as np
import os


logging.getLogger('matplotlib').setLevel(logging.ERROR)


logger = logging.getLogger(__name__)


class DaliAligner():
    DAT_PATH = "/home/iscb/wolfson/hagairavid/DaliLite.v5/DAT"
    IMPORT_PATH = '/home/iscb/wolfson/hagairavid/DaliLite.v5/bin/import.pl'
    DALI_PATH = '/home/iscb/wolfson/hagairavid/DaliLite.v5/bin/dali.pl'
    def __init__(self) -> None:  
        # super().__init__()
        self.name = "DaliAligner"

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

    def impose_structure(self, ref_protein, mov_protein, ligand_dir: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
       
        mov_name, mov_chain = mov_protein._pdb_name, mov_protein._chain_id
        ref_name, ref_chain = ref_protein._pdb_name, ref_protein._chain_id
        mov_path = os.path.join("ligand_alligner", ligand_dir, 'pdb' + mov_protein._pdb_name + '.ent')
        ref_path = os.path.join("ligand_alligner", ligand_dir, 'pdb' + ref_name + '.ent')
        try:
            # temp_dir = f'temp/{mov_name}_{ref_name}'
            # os.makedirs(temp_dir, exist_ok=True)
            os.chdir(os.path.join("ligand_alligner", ligand_dir))
            import_1 = subprocess.run([self.IMPORT_PATH, '--pdbfile', mov_path, '--pdbid', mov_name, '--dat', self.DAT_PATH], capture_output=True, text=True, check=True)
            import_2 = subprocess.run([self.IMPORT_PATH, '--pdbfile', ref_path, '--pdbid', ref_name, '--dat', self.DAT_PATH], capture_output=True, text=True, check=True)
            allign_log = subprocess.run([self.DALI_PATH, '--cd1', ref_name + ref_chain , '--cd2', mov_name + mov_chain,
                                          '--dat1', self.DAT_PATH, '--dat2', self.DAT_PATH, '--title',
                                            "output options" ,'--outfmt', "summary,alignments,equivalences,transrot", "--clean"
                                              ], capture_output=True, text=True, check=True)

            matrix, rmsd, _ = DaliAligner.extract_matrices_combined(f'{ref_name}{ref_chain}.txt')
            try:
                os.remove(ref_name + '.dssp')
                os.remove(mov_name + '.dssp')
                # os.remove(ref_name + ref_chain + '.txt')
            except OSError:
                pass
  
            os.chdir('/home/iscb/wolfson/hagairavid/ligand_alligner')
            if matrix is not None:
                R = np.linalg.inv(matrix[:, :3])
                t = matrix[:, 3]
                return [R], [t], [rmsd], []
            else:
                return [], [], [], []
            
        except Exception as e:
            print(e)
            return [], [], [], []