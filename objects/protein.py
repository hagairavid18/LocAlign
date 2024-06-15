import os
import logging
import warnings
from scipy.spatial.distance import cdist

from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB import PDBList, MMCIFParser
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Atom import Atom
import numpy as np
from Bio.PDB.Polypeptide import protein_letters_3to1

logger = logging.getLogger(__name__)


warnings.filterwarnings("ignore", category=PDBConstructionWarning)


class Protein:
    def __init__(self, pdb_name: str, chain_id: str, ligand_name: str, model_idx: int = 0) -> None:
  
        self._pdb_name = pdb_name
        self._chain_id = chain_id
        self._model_idx = model_idx
        self._ligand_name = ligand_name
        self._structure: Structure = self._init_structure()
        self._ligand_model, self._num_of_ligand_atoms = self._save_ligand_model()
    
    def _init_structure(self) -> Structure:

        pdb_list = PDBList(verbose=False)
        pdb_file_path = pdb_list.retrieve_pdb_file(self._pdb_name, pdir=f'alligned_structures/{self._ligand_name}', file_format='mmCif')
        _ = pdb_list.retrieve_pdb_file(self._pdb_name, pdir=f'alligned_structures/{self._ligand_name}', file_format='pdb')
        mmcif_parser = MMCIFParser()
        try:
            structure: Structure = mmcif_parser.get_structure(self._pdb_name, pdb_file_path)
        except FileNotFoundError:
            logger.info(f"Could not find the structre of {self._pdb_name}. Please remove this query")
            structure = Structure(self._pdb_name)
        return structure
  
    def _save_ligand_model(self) -> int:
        peptide_model = self.get_model(self._model_idx)
        ligand_model, num_of_ligand_atoms = Protein.create_ligand_model(peptide_model, self._ligand_name, self._chain_id)
        save_dir = f'alligned_structures/{self._ligand_name}'
        io = PDBIO()
        io.set_structure(ligand_model)
        io.save(os.path.join(save_dir, f"{self._pdb_name}_ligand.pdb"))
        return ligand_model, num_of_ligand_atoms
    
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
    
    def get_pocket_atoms(self, residues_thresh: float = 5.0, atoms_thresh: float =  8.0, ligand_res_idx: int = 0) -> list[Atom]:
        ligand_residue = self.get_ligand_residues()[ligand_res_idx] # TODO: handle ligand with more residues
        ligand_coors = [atom.coord for atom in ligand_residue.get_atoms() if atom.element != "H"]
        pocket_residues = []
        pocket_atoms = []
        for residue in list(self.get_model(self._model_idx, True)):
            res_coors = [atom.coord for atom in residue.get_atoms() if atom.element != "H"]
            distances = cdist(ligand_coors, res_coors, metric='euclidean')
            # min_dist_idx = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
            # min_distance = distances[min_dist_idx]
            # 
            if distances.min() < residues_thresh:
                pocket_residues.append(residue)
                pocket_atoms += list(residue)
        
        pocket_atoms_coors = np.array([atom.coord for atom in pocket_atoms if atom.element != "H"])
        close_atoms = np.empty((0, 3))

        for residue in list(self.get_model(self._model_idx, True)):
            res_coors = np.array([atom.coord for atom in residue.get_atoms() if atom.element != "H"])
            distances = cdist(pocket_atoms_coors, res_coors, metric='euclidean')
            close_atoms = np.vstack((close_atoms, res_coors[np.unique(np.where(distances < atoms_thresh)[1])]))
    
        return np.vstack((close_atoms, pocket_atoms_coors))
    
    @staticmethod
    def create_ligand_model(model: Model, ligand_name: str, chain_idx: int) -> tuple[Model, list[int]]:
        
        ligand_model = Model(model.id)
        chain: Chain = model[chain_idx]
        ligand_chain = Chain(chain.id)
        for residue in list(chain):
            if residue.resname == ligand_name:
                heavy_residue = Residue(residue.id, residue.resname, residue.get_segid())
                for atom in residue.get_atoms():
                    if atom.element != "H":
                        heavy_residue.add(atom)
                ligand_chain.add(heavy_residue)

        num_atoms = [len(residue) for residue in ligand_chain]
        logger.debug(f"Ligand has {num_atoms} atoms")
        ligand_model.add(ligand_chain)
        return ligand_model, num_atoms
    
    def _get_atoms(self, id: str = " ") -> list[list[Atom]]:
        atoms =  [] 

        for residue in list(self.get_model(self._model_idx, True)):
            if residue.resname == id:
                atoms.append(list(residue))
     
        return atoms
    
    @staticmethod
    def get_residue_data(chain: Chain) -> tuple[np.ndarray, str]:
        coords = []
        seq = []
        for residue in chain.get_residues():
            if "CA" in residue.child_dict and residue.resname in protein_letters_3to1:
                coords.append(residue.child_dict["CA"].coord)
                seq.append(protein_letters_3to1[residue.resname])

        return np.vstack(coords), "".join(seq)
    
    def get_num_of_ligand_atoms(self) -> int:
        return self._num_of_ligand_atoms


  





