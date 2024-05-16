import os
import logging
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB import PDBList, MMCIFParser
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Atom import Atom

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
    
    def get_ligand_atoms(self, ligand_name: int = 0) -> list[Atom]:
        return list(list(self._ligand_model.get_chains())[0])
    
    staticmethod
    def create_ligand_model(model: Model, ligand_name: str, chain_idx: int) -> tuple[Model, list[int]]:
        
        ligand_model = Model(model.id)
        chain: Chain = model[chain_idx]
        ligand_chain = Chain(chain.id)
        for residue in list(chain):
            if residue.resname == ligand_name: # TODO: There is attribute residue.resname which returns the ligand name itself.
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
    
    def get_num_of_ligand_atoms(self) -> int:
        return self._num_of_ligand_atoms


  





