import os
import logging
import pickle,gzip
import warnings
# from scipy.spatial.distance import cdist
import subprocess


from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Polypeptide import is_aa


from Bio.PDB.Atom import Atom
import numpy as np
# from Bio.PDB.Polypeptide import protein_letters_3to1

from miners.utils.constants import LIGAND_DIR


logger = logging.getLogger(__name__)

protein_letters_3to1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
                  'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
                  'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M'}


warnings.filterwarnings("ignore", category=PDBConstructionWarning)


def is_PDB_identifier(identifier: str) -> bool:
    """Check if identifier is a valid PDB ID (4 alphanumeric characters)."""
    return (len(identifier) == 4) and identifier.isalnum()


def is_UniProt_identifier(identifier: str) -> bool:
    """Check if identifier is a valid UniProt ID (6 or 10 characters with specific pattern)."""
    L = len(identifier)
    correct_length = L in [6, 10]
    if not correct_length:
        return False
    
    only_alnum = identifier.isalnum()
    only_upper = (identifier.upper() == identifier)
    first_is_letter = identifier[0].isalpha()
    six_is_digit = identifier[5].isnumeric()
    
    valid_uniprot_id = correct_length and only_alnum and only_upper and first_is_letter and six_is_digit
    
    if L == 10:
        seven_is_letter = identifier[6].isalpha()
        last_is_digit = identifier[1].isnumeric()
        valid_uniprot_id = valid_uniprot_id and seven_is_letter and last_is_digit
    
    return valid_uniprot_id


class Protein:
    def __init__(self, pdb_name: str, chain_id: str, ligand_name: str, model_idx: int = 0, save_models: bool = True, ligand_dir: str = LIGAND_DIR, pdb_id: str | None = None) -> None:
  
        self._pdb_name = pdb_name  # keep original (path or ID) for local loading
        self._pdb_id = pdb_id or pdb_name  # use provided ID (pre-hashed if path) or fall back to name
        self._chain_id = chain_id
        self._model_idx = model_idx
        self._ligand_name = ligand_name
        self._ligand_dir = ligand_dir
        self._structure: Structure = self._init_structure()
        self._ligand_model, self._num_of_ligand_atoms = self._get_ligand_model(save_models)
        self.__non_ligand_model, self._num_of_non_ligand_atoms = self._get_non_ligand_model(save_models)

    @staticmethod
    def _make_file_key(pdb_id: str) -> str:
        """Filesystem-friendly name from the ID (which may be hashed already)."""
        return os.path.basename(pdb_id)
    
    def _init_structure(self) -> Structure:
        cache_dir = f"{self._ligand_dir}/{self._ligand_name}/cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{self._pdb_id}.pkl.gz")

        # Check if cached structure exists
        if os.path.exists(cache_file):
            try:
                with gzip.open(cache_file, 'rb') as f:
                    logger.info(f"Loading structure {self._pdb_id} from cache.")
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached structure for {self._pdb_id}: {e}")
        
        # First, allow direct local file paths (pdb or cif)
        if os.path.isfile(self._pdb_name):
            local_path = self._pdb_name
            try:
                if local_path.lower().endswith('.cif'):
                    parser = MMCIFParser()
                else:
                    parser = PDBParser(QUIET=True)
                structure = parser.get_structure(self._pdb_id, local_path)
                logger.info(f"Loaded local structure from {local_path}")
            except Exception as e:
                logger.error(f"Failed to parse local structure {local_path}: {e}")
                structure = Structure(self._pdb_id)
            # Cache the loaded structure
            try:
                with gzip.open(cache_file, 'wb') as f:
                    pickle.dump(structure, f)
            except Exception as e:
                logger.warning(f"Failed to cache local structure {self._pdb_id}: {e}")
            return structure

        # Determine if it's a PDB ID or UniProt ID
        is_pdb = is_PDB_identifier(self._pdb_id)
        is_uniprot = is_UniProt_identifier(self._pdb_id)
        
        if not (is_pdb or is_uniprot):
            logger.error(f"Identifier {self._pdb_id} is neither a valid PDB nor UniProt ID")
            return Structure(self._pdb_id)
        
        # Parse the structure if not cached
        mmcif_file_path = f"{self._ligand_dir}/{self._ligand_name}/{self._pdb_id}.cif"
        
        if is_pdb:
            url = f"https://files.rcsb.org/download/{self._pdb_id}.cif"
        else:  # is_uniprot
            alphafold_server = 'https://alphafold.ebi.ac.uk/files'
            url = f"{alphafold_server}/AF-{self._pdb_id}-F1-model_v6.cif"

        # Run wget command to download the file
        try:
            subprocess.run(["wget", url, "-O", mmcif_file_path, "--quiet"], check=True, timeout=60)
            if is_pdb:
                logger.info(f"Successfully downloaded PDB structure {self._pdb_id}.cif from RCSB")
            else:
                logger.info(f"Successfully downloaded AlphaFold structure for UniProt {self._pdb_id}.cif from AFDB")
        except subprocess.CalledProcessError:
            if is_pdb:
                logger.error(f"Failed to download {self._pdb_id}.cif from {url}")
            else:
                logger.error(f"Failed to download AlphaFold structure for {self._pdb_id} from {url}")
            structure = Structure(self._pdb_id)  # Return empty structure if download fails
            return structure
        except subprocess.TimeoutExpired:
            logger.error(f"Download timeout for {self._pdb_id} from {url}")
            structure = Structure(self._pdb_id)
            return structure

        # Parse the downloaded mmCIF file
        try:
            mmcif_parser = MMCIFParser()
            structure = mmcif_parser.get_structure(self._pdb_id, mmcif_file_path)
        except Exception as e:
            logger.error(f"Failed to parse the downloaded structure {self._pdb_id}: {e}")
            structure = Structure(self._pdb_id)  # Return empty structure on parse failure
        
        # Save the parsed structure to the cache
        try:
            with gzip.open(cache_file, 'wb') as f:
                pickle.dump(structure, f)
                logger.info(f"Cached structure {self._pdb_id} to {cache_file}.")
        except Exception as e:
            logger.warning(f"Failed to cache structure {self._pdb_id}: {e}")
        
        return structure

    def _get_ligand_model(self, save: bool = True) -> int:
        save_dir = f'{self._ligand_dir}/{self._ligand_name}'
        save_path = os.path.join(save_dir, f"{self._pdb_id}{self._chain_id}_ligand.pdb")
        peptide_model = self.get_model(self._model_idx)
        ligand_model, num_of_ligand_atoms = Protein.create_ligand_model(peptide_model, self._ligand_name, self._chain_id)
        if save and not os.path.exists(save_path):
            io = PDBIO()
            io.set_structure(ligand_model)
            io.save(save_path)
        return ligand_model, num_of_ligand_atoms
    
    def _get_non_ligand_model(self, save: bool = True) -> tuple[Model, int]:
        # Define cache directory and file
        save_dir = f"{self._ligand_dir}/{self._ligand_name}"
        save_path = os.path.join(save_dir, f"{self._pdb_id}{self._chain_id}_non_ligand_.ent")

        peptide_model = self.get_model(self._model_idx)
        non_ligand_model, num_of_ligand_atoms = Protein.create_non_ligand_model(peptide_model, self._chain_id)

        if save and not os.path.exists(save_path):
            os.makedirs(save_dir, exist_ok=True)
            io = PDBIO()
            io.set_structure(non_ligand_model)
            io.save(save_path)

        return non_ligand_model, sum(num_of_ligand_atoms)
    
    def get_model(self, model_idx: int, only_chain: bool = False) -> Model | Chain:
        if only_chain:
            return self._structure[model_idx][self._chain_id]
        
        return self._structure[model_idx]
    
    def save_chain_model(self, model_idx: int, only_chain: bool = False) -> Model | Chain:
        if only_chain:
            return self._structure[model_idx][self._chain_id]
        
        return self._structure[model_idx]
    
    def get_ligand_residues(self) -> list[Residue]:
        return list(list(self._ligand_model.get_chains())[0])
    
    # def get_pocket_atoms_within_4A(self, distance_thresh: float = 4.0, ligand_res_idx: int = 0) -> np.ndarray:
        
    #     ligand_residue = self.get_ligand_residues()[ligand_res_idx]  # Handle ligand with more residues if needed
    #     ligand_coors = np.array([atom.coord for atom in ligand_residue.get_atoms() if atom.element != "H"])
    #     pocket_atoms = []
    #     pocket_residue_indices = []

    #     # First loop: Identify atoms within the threshold distance
    #     for residue in list(self.get_model(self._model_idx, True)):
    #         if residue.resname == self._ligand_name:  # Skip ligand itself
    #             continue
    #         for atom in residue.get_atoms():
    #             if atom.element != "H":  # Ignore hydrogens
    #                 atom_coor = np.array([atom.coord])
    #                 distances = cdist(ligand_coors, atom_coor, metric='euclidean')
    #                 if distances.min() < distance_thresh:
    #                     pocket_atoms.append(atom)
    #                     pocket_residue_indices.append(residue.id[1])

    #     # Extract coordinates of pocket atoms
    #     pocket_atoms_coors = np.array([atom.coord for atom in pocket_atoms])

    #     return pocket_atoms_coors, pocket_residue_indices
    
    @staticmethod
    def get_atoms_within_distance(all_atoms_coord: np.ndarray, center: np.ndarray, distance_thresh: float = 4.0) -> np.ndarray:
        """
        Extract atoms within a specified distance from the given center point.
        
        Args:
            center: A 3D coordinate representing the tarerence point.
            distance_thresh: The threshold distance to include atoms.
            
        Returns:
            An array of coordinates for atoms within the specified distance.
        """
        atoms_within_distance = []
        for atom_coord in all_atoms_coord:
            distance = np.linalg.norm(atom_coord - center)
            if distance < distance_thresh:
                atoms_within_distance.append(atom_coord)
        return np.array(atoms_within_distance)
    
    def get_pocket_atoms(
        self, 
        residues_thresh: float = 5.0, 
        atoms_thresh: float = 8.0, 
        ligand_res_idx: int = 0, 
        first_loop_only: bool = False
    ) -> np.ndarray:
        ligand_residue = self.get_ligand_residues()[ligand_res_idx]  # TODO: handle ligand with more residues
        ligand_coors = [atom.coord for atom in ligand_residue.get_atoms() if atom.element != "H"]
        pocket_residues = []
        pocket_atoms = []
        pocket_residue_indices = []
        for residue in list(self.get_model(self._model_idx, True)):
            if residue.resname == self._ligand_name:
                continue
            res_coors = [atom.coord for atom in residue.get_atoms() if atom.element != "H"]
            distances = cdist(ligand_coors, res_coors, metric='euclidean')
            if distances.min() < residues_thresh:
                pocket_residues.append(residue)
                pocket_atoms += list(residue)
                pocket_residue_indices.append(residue.id[1])
        
        # If only results from the first loop are needed
        if first_loop_only:
            pocket_atoms_coors = np.array([atom.coord for atom in pocket_atoms if atom.element != "H"])
            return pocket_atoms_coors, pocket_residue_indices

        # Continue with second loop
        pocket_atoms_coors = np.array([atom.coord for atom in pocket_atoms if atom.element != "H"])
        pocket_atoms_ca_coors = np.array([atom.coord for atom in pocket_atoms if atom.element != "CA"])
        close_atoms = np.empty((0, 3))

        for residue in list(self.get_model(self._model_idx, True)):
            if residue.resname == self._ligand_name:
                continue
            if residue in pocket_residues:
                continue
            res_coors = np.array([atom.coord for atom in residue.get_atoms() if atom.name == "CA"])
            if res_coors.any():
                distances = cdist(pocket_atoms_coors, res_coors, metric='euclidean')
                close_atoms = np.vstack((close_atoms, res_coors[np.unique(np.where(distances < atoms_thresh)[1])]))
                close_ca_indices = np.unique(np.where(distances < atoms_thresh)[1])
                if len(close_ca_indices) > 0:
                    pocket_residue_indices.append(residue.id[1])
                    pocket_residues

        result = np.vstack((close_atoms, pocket_atoms_coors))

        return result, pocket_residue_indices

    
    def get_pocket_atoms_with_ca(
    self, residues_thresh: float = 5.0, atoms_thresh: float = 8.0, ligand_res_idx: int = 0
) -> tuple[np.ndarray, list[int]]:
        """
        Calculate pocket atoms and return only the Cα coordinates of the pocket along with residue indices.
        """
        # Get ligand residue and its coordinates
        ligand_residue = self.get_ligand_residues()[ligand_res_idx]
        ligand_coors = [atom.coord for atom in ligand_residue.get_atoms() if atom.element != "H"]

        pocket_residues = []
        pocket_atoms = []
        pocket_ca_residue_indices = []
        pocket_ca_coords = []

        # First loop: Find residues near the ligand (within residues_thresh)
        for residue in list(self.get_model(self._model_idx, True)):
            if residue.resname == self._ligand_name:
                continue

            # Get all heavy atom coordinates in the residue
            res_coors = [atom.coord for atom in residue.get_atoms() if atom.element != "H"]
            distances = cdist(ligand_coors, res_coors, metric="euclidean")
            if distances.min() < residues_thresh:
                pocket_residues.append(residue)
                pocket_atoms += list(residue)

                # Add Cα atom coordinates if available
                ca_atom = next((atom for atom in residue.get_atoms() if atom.name == "CA"), None)
                if ca_atom is not None:
                    pocket_ca_coords.append(ca_atom.coord)
                    pocket_ca_residue_indices.append(residue.id[1])

        pocket_atoms_coors = np.array([atom.coord for atom in pocket_atoms if atom.element != "H"])
        pocket_ca_coords = np.array(pocket_ca_coords)

        # Second loop: Include residues with Cα atoms near the pocket (within atoms_thresh)
        for residue in list(self.get_model(self._model_idx, True)):
            if residue.resname == self._ligand_name or residue in pocket_residues:
                continue

            # Check distances for Cα atoms
            ca_atom = next((atom for atom in residue.get_atoms() if atom.name == "CA"), None)
            if ca_atom is not None:
                ca_coord = ca_atom.coord
                distances = cdist(pocket_atoms_coors, np.array([ca_coord]), metric="euclidean")
                if distances.min() < atoms_thresh:
                    pocket_ca_coords = np.vstack((pocket_ca_coords, ca_coord))
                    pocket_ca_residue_indices.append(residue.id[1])

        # Ensure consistency in return values
        pocket_ca_coords = np.array(pocket_ca_coords)
        return pocket_ca_coords, pocket_ca_residue_indices

    def get_pocket_residue_ids(
        self,
        distance_thresh: float = 4.0,
        ligand_res_idx: int = 0,
    ) -> list[int]:
        """
        Return residue IDs for all residues whose heavy atoms are within
        ``distance_thresh`` Å of the ligand heavy atoms.

        Args:
            distance_thresh: Maximum distance in Å to consider a residue part of the pocket (default: 4.0).
            ligand_res_idx: Index of the ligand residue to use (default: 0).

        Returns:
            List of residue sequence numbers (Bio.PDB residue.id[1]) that define the pocket.
        """
        try:
            ligand_residue = self.get_ligand_residues()[ligand_res_idx]
        except Exception:
            return []

        ligand_coors = np.array([
            atom.coord for atom in ligand_residue.get_atoms() if getattr(atom, "element", None) != "H"
        ], dtype=float)

        if ligand_coors.size == 0:
            return []

        pocket_residue_ids: list[int] = []

        for residue in list(self.get_model(self._model_idx, True)):
            # Skip ligand residues themselves
            if residue.resname == self._ligand_name:
                continue

            res_coors = np.array([
                atom.coord for atom in residue.get_atoms() if getattr(atom, "element", None) != "H"
            ], dtype=float)

            if res_coors.size == 0:
                continue

            # Compute pairwise distances between ligand heavy atoms and residue heavy atoms
            # Using numpy broadcasting to avoid scipy dependency
            diff = ligand_coors[:, None, :] - res_coors[None, :, :]
            dists = np.linalg.norm(diff, axis=-1)

            if np.min(dists) < distance_thresh:
                pocket_residue_ids.append(residue.id[1])

        return pocket_residue_ids

    @staticmethod
    def create_ligand_model(model: Model, ligand_name: str, chain_idx: int) -> tuple[Model, list[int]]:
        
        ligand_model = Model(model.id)
        chain: Chain = model[chain_idx]
        ligand_chain = Chain(chain.id)
        for residue in list(chain):
            if residue.resname == ligand_name:
                heavy_residue = Residue(residue.id, residue.resname[:3], residue.get_segid())
                for atom in residue.get_atoms():
                    if atom.element != "H":
                        heavy_residue.add(atom)
                ligand_chain.add(heavy_residue)

        num_atoms = [len(residue) for residue in ligand_chain]
        logger.debug(f"Ligand has {num_atoms} atoms")
        ligand_model.add(ligand_chain)
        if len(ligand_model) == 0:
            raise ValueError(f"Ligand {ligand_name} not found in the model.")
        return ligand_model, num_atoms
   
    @staticmethod
    def create_non_ligand_model(model: Model, chain_idx: int) -> tuple[Model, list[int]]:
        """
        Build a new Model with a single chain that contains only heavy-atom
        standard amino-acid residues from the given chain. All ligands,
        waters, ions, and other hetero residues are excluded.
        """
        protein_model = Model(model.id)
        chain: Chain = model[chain_idx]
        protein_chain = Chain(chain.id)

        kept_res, dropped_res = 0, 0
        for residue in chain:
            # Keep only standard amino acids (drops ligands/waters/ions etc.)
            if is_aa(residue, standard=False):
                new_res = Residue(residue.id, residue.resname, residue.get_segid())
                for atom in residue.get_atoms():
                    # keep heavy atoms only
                    if getattr(atom, "element", None) != "H":
                        new_res.add(atom)
                protein_chain.add(new_res)
                kept_res += 1
            else:
                dropped_res += 1

        num_atoms = [len(res) for res in protein_chain]
        logger.info(f"Kept {kept_res} AA residues; dropped {dropped_res} non-AA residues. "
                     f"Atoms per kept residue: {num_atoms}")

        protein_model.add(protein_chain)
        return protein_model, num_atoms
    
    def _get_atoms(self, id: str = " ", all_atoms: bool = False) -> list[list[Atom]]:
        atoms =  [] 

        for residue in list(self.get_model(self._model_idx, True)):
            if all_atoms or residue.resname == id:
                atoms += (list(residue))
        return atoms
    
    @staticmethod
    def save_coordinates_to_pdb(coordinates, output_filename):
        """
        Saves a list of 3D coordinates to a PDB file using Biopython.

        Parameters:
        coordinates (np.ndarray): A numpy array of shape (N, 3) containing the 3D coordinates.
        output_filename (str): The name of the output PDB file.

        """
        structure = Structure("example_structure")
        model = Model(0)
        chain = Chain("A")
        for i, coord in enumerate(coordinates):
            x, y, z = coord
            residue = Residue((" ", i + 1, " "), "ALA", " ")
            atom = Atom("CA", coord, 1.0, 1.0, " ",  "CA", i + 1, "C")
            residue.add(atom)
            chain.add(residue)
        model.add(chain)
        structure.add(model)
        io = PDBIO()
        io.set_structure(structure)
        io.save(output_filename)
    
    @staticmethod
    def get_residue_data(chain: Chain) -> tuple[np.ndarray, str, list[int]]:
        """
        Extracts CA coordinates, residue sequence, and residue indices from a chain.

        Args:
            chain (Bio.PDB.Chain): A chain from a PDB structure.

        Returns:
            tuple[np.ndarray, str, list[int]]: A tuple containing:
                - A NumPy array of CA atom coordinates.
                - A string of the protein sequence derived from the residues.
                - A list of residue indices corresponding to the coordinates.
        """
        coords = []
        seq = []
        residue_indices = []
        
        for residue in chain.get_residues():
            if "CA" in residue.child_dict and residue.resname in protein_letters_3to1:
                coords.append(residue.child_dict["CA"].coord)
                seq.append(protein_letters_3to1[residue.resname])
                residue_indices.append(residue.id[1])  # Extract residue index

        return np.vstack(coords), "".join(seq), residue_indices

    
    def get_num_of_ligand_atoms(self) -> int:
        return self._num_of_ligand_atoms
