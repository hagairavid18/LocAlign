import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB import PDBList, MMCIFParser
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Structure import Structure
from Bio.PDB.Atom import Atom

warnings.filterwarnings("ignore", category=PDBConstructionWarning)


class Protein:
    def __init__(self, pdb_name: str, chain_id: str, ligand_name: str, model_idx: int = 0) -> None:
  
        self._pdb_name = pdb_name
        self._chain_id = chain_id
        self._model_idx = model_idx
        self._ligand_name = ligand_name
        self._structure: Structure = self._init_structure()
    
    def _init_structure(self) -> Structure:

        pdb_list = PDBList()
        pdb_file_path = pdb_list.retrieve_pdb_file(self._pdb_name, pdir=f'alligned_structures/{self._ligand_name}', file_format='mmCif')
        mmcif_parser = MMCIFParser()
        structure: Structure = mmcif_parser.get_structure(self._pdb_name, pdb_file_path)
        return structure
  
    def get_model(self, model_idx: int, only_chain: bool = False) -> Model | Chain:
        if only_chain:
            return self._structure[model_idx][self._chain_id]
        
        return self._structure[model_idx]
    
    def get_ligand_atoms(self, ligand_id_name: int = 0) -> list[Atom]:
        return self._get_atoms(ligand_id_name)
    
    def _get_atoms(self, id: str = " ") -> list[Atom]:
        atoms =  [] 

        for residue in list(self.get_model(self._model_idx, True)):
            if residue.id[0] == id:
                atoms += list(residue)
     
        return atoms
    


  





