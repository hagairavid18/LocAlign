import logging
import os
import subprocess
import tempfile
import numpy as np
from Bio.PDB import PDBParser, PDBIO

logging.getLogger('matplotlib').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class APOCAligner:
    def __init__(self, apoc_path: str = None, fpocket_path: str = None) -> None:
        self.name = "APOCAligner"

        # Find APOC executable
        if apoc_path:
            self.apoc_path = apoc_path
        else:
            paths_to_try = [
                os.path.expanduser("~/LocAlign/apoc/bin/apoc"),
                "/home/iscb/wolfson/hagairavid/LocAlign/apoc/bin/apoc",
                "apoc",
            ]
            self.apoc_path = None
            for path in paths_to_try:
                if os.path.exists(path):
                    self.apoc_path = path
                    break
                if subprocess.run(["which", path], capture_output=True).returncode == 0:
                    self.apoc_path = path
                    break

            if not self.apoc_path:
                raise FileNotFoundError("APOC executable not found. Please install it or provide the path.")

        # Find fpocket executable
        if fpocket_path:
            self.fpocket_path = fpocket_path
        else:
            if subprocess.run(["which", "fpocket"], capture_output=True).returncode == 0:
                self.fpocket_path = "fpocket"
            else:
                raise FileNotFoundError("fpocket executable not found. Please install it via conda.")

        logger.info(f"Using APOC at: {self.apoc_path}")
        logger.info(f"Using fpocket at: {self.fpocket_path}")

    def _get_structure_path(self, protein, ligand_dir: str) -> str:
        """Extract non-ligand PDB/CIF file path from Protein object."""
        pdb_name = protein._pdb_name
        chain_id = protein._chain_id
        
        # Use non-ligand structure file
        non_ligand_base = f"{pdb_name}{chain_id}_non_ligand_"
        
        # Try .ent first (most common for non-ligand structures)
        ent_path = os.path.join(ligand_dir, non_ligand_base + ".ent")
        if os.path.exists(ent_path):
            return ent_path
        
    def _clean_pdb_for_fpocket(self, pdb_file: str, dest_path: str) -> bool:
        """
        Create a fpocket-friendly PDB by keeping only ATOM/HETATM records.
        Returns True on success.
        """
        try:
            wrote_any = False
            with open(pdb_file, 'r') as inp, open(dest_path, 'w') as out:
                for line in inp:
                    if line.startswith('ATOM') or line.startswith('HETATM'):
                        out.write(line)
                        wrote_any = True
            return wrote_any
        except Exception as e:
            logger.error(f"Error cleaning PDB for fpocket: {e}")
            return False

    def _extract_pockets_fpocket(self, pdb_file: str, temp_dir: str) -> str:
        """
        Extract pockets from a PDB file using fpocket.
        
        Returns:
            Path to directory containing fpocket results
        """
        try:
            # Prepare a cleaned PDB inside temp_dir to avoid path issues
            base = os.path.splitext(os.path.basename(pdb_file))[0]
            cleaned_pdb = os.path.join(temp_dir, f"{base}_clean.pdb")

            ok = self._clean_pdb_for_fpocket(pdb_file, cleaned_pdb)
            if not ok:
                logger.warning(f"No ATOM/HETATM lines found in {pdb_file}. Skipping fpocket.")
                return None

            # Run fpocket (without -d). It will create <basename>_out/ with pockets inside.
            cmd = [self.fpocket_path, "-f", cleaned_pdb]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_dir)

            if result.returncode != 0:
                logger.warning(f"fpocket extraction failed for {pdb_file}: {result.stderr or result.stdout}")
                return None

            out_dir = os.path.join(temp_dir, f"{os.path.splitext(os.path.basename(cleaned_pdb))[0]}_out")
            pockets_dir = os.path.join(out_dir, "pockets")
            if not os.path.isdir(pockets_dir):
                logger.warning(f"fpocket output pockets directory not found: {pockets_dir}")
                return None

            return pockets_dir

        except Exception as e:
            logger.error(f"Error extracting pockets with fpocket: {e}")
            return None

    def _create_apoc_input(self, pdb_file: str, pocket_dir: str, temp_file: str) -> bool:
        """
        Create APOC-compatible input file with full chain + top 10 pockets.
        
        Format:
        <Full chain PDB coordinates>
        TER
        PKT  <num pocket residues> <pocket volume> <pocket name>
        <pocket PDB coordinates>
        TER
        """
        try:
            # Write full chain coordinates
            with open(temp_file, 'w') as out_f:
                for line in open(pdb_file):
                    if line.startswith(('ATOM', 'HETATM')):
                        out_f.write(line)
                out_f.write("TER\n")
            
            # Parse pocket volumes from fpocket output
            if pocket_dir and os.path.exists(pocket_dir):
                volumes_map: dict[int, float] = {}
                out_dir = os.path.dirname(pocket_dir)
                info_files = [f for f in os.listdir(out_dir) if f.endswith('_info.txt')]
                info_path = os.path.join(out_dir, info_files[0]) if info_files else None
                    
                if info_path and os.path.isfile(info_path):
                    try:
                        with open(info_path, 'r') as inf:
                            current_pocket = None
                            for line in inf:
                                if 'Pocket' in line and ':' in line:
                                    try:
                                        parts = line.split()
                                        for i, part in enumerate(parts):
                                            if part == 'Pocket' and i + 1 < len(parts):
                                                current_pocket = int(parts[i + 1])
                                                break
                                    except (IndexError, ValueError):
                                        current_pocket = None
                                elif current_pocket is not None and 'Volume' in line:
                                    try:
                                        vol_str = line.split(':')[-1].strip().split()[0]
                                        volumes_map[current_pocket] = float(vol_str)
                                    except (IndexError, ValueError):
                                        pass
                    except Exception as e:
                        logger.debug(f"Could not parse fpocket info file: {e}")

                # Get top 10 pockets (fpocket ranks by score: pocket1 = best)
                all_pocket_files = sorted(
                    [f for f in os.listdir(pocket_dir) if f.startswith('pocket') and f.endswith('.pdb')],
                    key=lambda f: int(''.join(c for c in f if c.isdigit())) if any(c.isdigit() for c in f) else float('inf')
                )
                pocket_files = all_pocket_files[:10]
                
                for pocket_file in pocket_files:
                    pocket_path = os.path.join(pocket_dir, pocket_file)
                    pocket_name = pocket_file.replace('.pdb', '')
                    
                    # Extract pocket index for volume lookup
                    pocket_idx = int(''.join(ch for ch in pocket_name if ch.isdigit())) if any(ch.isdigit() for ch in pocket_name) else None
                    
                    # Filter to CA/CB atoms only (required by APOC)
                    pocket_atoms = []
                    with open(pocket_path) as f:
                        for line in f:
                            if line.startswith(('ATOM', 'HETATM')):
                                atom_name = line[12:16].strip()
                                if atom_name in ['CA', 'CB']:
                                    pocket_atoms.append(line)
                    
                    if pocket_atoms:
                        # Count unique residues
                        residue_set = {line[22:27].strip() for line in pocket_atoms}
                        num_residues = len(residue_set)
                        
                        # Scale fpocket volume (Ų) to APOC grid points
                        volume_angstrom = float(volumes_map.get(pocket_idx, 100.0))
                        volume_grid_pts = volume_angstrom * 50.0
                        
                        with open(temp_file, 'a') as out_f:
                            out_f.write(f"PKT  {num_residues:8d} {volume_grid_pts:12.2f} {pocket_name}\n")
                            for atom_line in pocket_atoms:
                                out_f.write(atom_line)
                            out_f.write("TER\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating APOC input file: {e}")
            return False

    @staticmethod
    def _parse_apoc_output(output: str) -> dict:
        """
        Extract best alignment from APOC output.
        
        Returns:
            Alignment dict containing:
            - type: 'global' or 'pocket'
            - rmsd: float
            - rotation: np.ndarray (3x3)
            - translation: np.ndarray (3,)
            - ps_score: float (pocket alignments only)
        
        APOC output format:
        PS-score = 4.32978, P-value = 0.2629E-060, Z-score =139.491
        RMSD =  1.86, Seq identity  = 0.333
        
         -------- rotation matrix to rotate Chain-1 to Chain-2 ------ 
         i          t(i)         u(i,1)         u(i,2)         u(i,3)
         1    101.3762504793   0.3225276405  -0.6602814686  -0.6782361708
         2    -46.5235517939   0.0107048431  -0.7139371685   0.7001279353
         3     37.9607422597  -0.9464995127  -0.2330710228  -0.2231962609
        """
        lines = output.splitlines()
        
        # Store all alignments (global + pockets)
        alignments = []
        current_alignment = {}
        
        try:
            i = 0
            while i < len(lines):
                line = lines[i]
                
                # Detect alignment type
                if "Global alignment" in line:
                    current_alignment = {'type': 'global'}
                elif "Pocket alignment" in line:
                    current_alignment = {'type': 'pocket'}
                
                # Extract PS-score (pocket alignments only)
                if "PS-score" in line and "=" in line:
                    try:
                        ps_str = line.split("=")[1].split(",")[0].strip()
                        current_alignment['ps_score'] = float(ps_str)
                    except (ValueError, IndexError):
                        pass
                
                # Extract RMSD
                if "RMSD" in line and "=" in line and current_alignment:
                    try:
                        rmsd_str = line.split("=")[1].split(",")[0].strip()
                        current_alignment['rmsd'] = float(rmsd_str)
                    except (ValueError, IndexError):
                        pass
                
                # Extract rotation matrix
                if "rotation matrix" in line.lower():
                    rot_rows = []
                    trans_vals = []
                    
                    # Skip header line and process next 3 lines
                    for j in range(1, 4):
                        if i + j + 1 < len(lines):
                            data_line = lines[i + j + 1].strip()
                            if data_line:
                                parts = data_line.split()
                                if len(parts) >= 5:
                                    try:
                                        trans_vals.append(float(parts[1]))
                                        rot_rows.append([
                                            float(parts[2]),
                                            float(parts[3]),
                                            float(parts[4])
                                        ])
                                    except (ValueError, IndexError):
                                        pass
                    
                    if len(rot_rows) == 3 and len(trans_vals) == 3:
                        current_alignment['rotation'] = np.array(rot_rows, dtype=np.float64)
                        current_alignment['translation'] = np.array(trans_vals, dtype=np.float64)
                        
                        # Save complete alignment
                        if 'rmsd' in current_alignment:
                            alignments.append(current_alignment.copy())
                            current_alignment = {}
                
                i += 1
            # Return best pocket if exists, else global
            pocket_aligns = [a for a in alignments if a.get('type') == 'pocket' and 'ps_score' in a]
            
            if pocket_aligns:
                return max(pocket_aligns, key=lambda a: a['ps_score'])
            
            global_aligns = [a for a in alignments if a.get('type') == 'global']
            return global_aligns[0] if global_aligns else None
            
        except Exception as e:
            logger.error(f"Error parsing APOC output: {e}")
            return None

    def impose_structure(self, tar_protein, src_protein, ligand_dir: str) -> tuple[list[np.ndarray], list[np.ndarray], float, list]:
        """
        Align src_protein to tar_protein using APOC with fpocket pocket extraction.
        
        Returns:
            Tuple of (rotation_matrices, translation_vectors, rmsd, [])
            - If pockets found: uses best pocket alignment by PS-score
            - Otherwise: uses global alignment
        """
        temp_dir = tempfile.mkdtemp()
        
        try:
            src_path = self._get_structure_path(src_protein, ligand_dir)
            tar_path = self._get_structure_path(tar_protein, ligand_dir)
            
            src_apoc_input = os.path.join(temp_dir, "src_apoc.pdb")
            tar_apoc_input = os.path.join(temp_dir, "tar_apoc.pdb")
            
            # Extract pockets
            src_pocket_dir = self._extract_pockets_fpocket(src_path, temp_dir)
            tar_pocket_dir = self._extract_pockets_fpocket(tar_path, temp_dir)
            
            # Create APOC inputs
            src_ok = self._create_apoc_input(src_path, src_pocket_dir, src_apoc_input)
            tar_ok = self._create_apoc_input(tar_path, tar_pocket_dir, tar_apoc_input)

            if not src_ok or not tar_ok:
                logger.error("Failed to create APOC input files")
                return [], [], None, []

            # Build APOC command with pocket parameters if both structures have pockets
            cmd = [self.apoc_path, src_apoc_input, tar_apoc_input]
            if src_pocket_dir and tar_pocket_dir:
                cmd.extend(["-pvol", "10", "-plen", "3"])

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_dir)

            if result.returncode != 0:
                logger.error(f"APOC failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                return [], [], None, []

            alignment = self._parse_apoc_output(result.stdout)

            if not alignment:
                logger.error("Failed to extract alignment from APOC output")
                return [], [], None, []

            # Log alignment type
            if alignment.get('type') == 'pocket':
                logger.info(f"Selected best pocket alignment: PS-score={alignment.get('ps_score'):.3f}, RMSD={alignment['rmsd']:.2f} Å")
            else:
                logger.info(f"Selected global alignment: RMSD={alignment['rmsd']:.2f} Å")
            
            return [alignment['rotation'].T], [alignment['translation']], alignment['rmsd'], []
            
        except Exception as e:
            logger.error(f"APOCAligner error: {e}")
            return [], [], None, []
        
        finally:
            # Cleanup temporary files
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
