import logging
import os
import subprocess
import tempfile
import numpy as np

logging.getLogger('matplotlib').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class USAligner:
    def __init__(self, usalign_path: str = None, alignment_mode: int = 5) -> None:
        self.name = "USAligner"
        self.alignment_mode = alignment_mode  # 0=monomeric (default), 5=fully non-sequential (fNS)

        # Find USalign executable
        if usalign_path:
            self.usalign_path = usalign_path
        else:
            paths_to_try = [
                os.path.expanduser("~/bin/USalign"),
                "USalign",
            ]
            self.usalign_path = None
            for path in paths_to_try:
                if os.path.exists(path):
                    self.usalign_path = path
                    break
                if subprocess.run(["which", path], capture_output=True).returncode == 0:
                    self.usalign_path = path
                    break

            if not self.usalign_path:
                raise FileNotFoundError("USalign executable not found. Please install it or provide the path.")

        logger.info(f"Using USalign at: {self.usalign_path} with alignment mode: {self.alignment_mode}")

    def _get_structure_path(self, protein, ligand_dir: str) -> str:
        """Extract non-ligand PDB/CIF file path from Protein object."""
        pdb_name = protein._pdb_name
        chain_id = protein._chain_id
        
        # Use non-ligand structure file (like DaliAligner does)
        non_ligand_base = f"{pdb_name}{chain_id}_non_ligand_"
        
        # Try .ent first (most common for non-ligand structures)
        ent_path = os.path.join(ligand_dir, non_ligand_base + ".ent")
        if os.path.exists(ent_path):
            return ent_path
        
        # Try .pdb
        pdb_path = os.path.join(ligand_dir, non_ligand_base + ".pdb")
        if os.path.exists(pdb_path):
            return pdb_path
        
        # Try .cif
        cif_path = os.path.join(ligand_dir, non_ligand_base + ".cif")
        if os.path.exists(cif_path):
            return cif_path
        
        raise FileNotFoundError(f"Non-ligand structure file not found for {pdb_name}{chain_id}")

    @staticmethod
    def _parse_usalign_output(output: str) -> tuple[float, np.ndarray, np.ndarray]:
        """Extract RMSD, rotation matrix, and translation vector from USalign stdout."""
        lines = output.splitlines()
        
        rmsd = None
        rotation = None
        translation = None
        
        for i, line in enumerate(lines):
            # Extract RMSD
            if "Aligned length" in line and "RMSD" in line:
                parts = line.split("RMSD=")
                if len(parts) > 1:
                    rmsd = float(parts[1].split(",")[0].strip())
            
            # Extract rotation matrix and translation vector from the combined table
            # Format:
            # ------ The rotation matrix to rotate Structure_1 to Structure_2 ------
            # m               t[m]        u[m][0]        u[m][1]        u[m][2]
            # 0      37.1798576239   0.4909157977   0.1882946688  -0.8506155402
            # 1     -10.6465666229  -0.3087919732   0.9505840198   0.0322108450
            # 2      13.1047016492   0.8146466700   0.2468504384   0.5248005947
            if "rotation matrix" in line.lower():
                rot_rows = []
                trans_vals = []
                # Skip the header line (next line after "rotation matrix")
                for j in range(2, 5):  # Read lines for indices 0, 1, 2
                    if i + j < len(lines):
                        parts = lines[i + j].strip().split()
                        if len(parts) >= 5:
                            try:
                                # parts[0] = m (row index)
                                # parts[1] = t[m] (translation component)
                                # parts[2-4] = u[m][0-2] (rotation matrix row)
                                trans_vals.append(float(parts[1]))
                                rot_rows.append([float(parts[2]), float(parts[3]), float(parts[4])])
                            except (ValueError, IndexError):
                                break
                
                if len(rot_rows) == 3 and len(trans_vals) == 3:
                    rotation = np.array(rot_rows)
                    translation = np.array(trans_vals)
        
        if rmsd is None or rotation is None or translation is None:
            raise ValueError(f"Failed to parse USalign output. RMSD: {rmsd}, Rotation: {rotation is not None}, Translation: {translation is not None}")
        
        return rmsd, rotation, translation

    def impose_structure(self, tar_protein, src_protein, ligand_dir: str, alignment_mode: int = 0) -> tuple[list[np.ndarray], list[np.ndarray], float, list]:
        """
        Align src_protein to tar_protein using USalign.
        
        Args:
            tar_protein: Target/fixed Protein object
            src_protein: Source/mobile Protein object
            ligand_dir: Ligand directory path
            alignment_mode: Alignment mode (0=monomeric, 5=fNS, 6=sNS)
        
        Returns:
            Tuple of (rotation_matrices, translation_vectors, rmsd, [])
        """
        try:
            # Get structure file paths from Protein objects
            src_path = self._get_structure_path(src_protein, ligand_dir)
            tar_path = self._get_structure_path(tar_protein, ligand_dir)
            
            # Run USalign with specified alignment mode
            # -mm: 0=monomeric (default), 5=fully non-sequential (fNS), 6=semi-non-sequential (sNS)
            # -m -: output rotation matrix to stdout
            cmd = [self.usalign_path, src_path, tar_path, "-mm", str(alignment_mode), "-m", "-"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"USalign failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                return [], [], None, []
            
            # Parse output to get RMSD and matrices directly from stdout
            rmsd, rotation, translation = self._parse_usalign_output(result.stdout)
            
            # USalign gives rotation from src to tar, return as-is
            return [rotation.T], [translation], rmsd, []
            
        except Exception as e:
            logger.error(f"USAligner error: {e}")
            return [], [], None, []
