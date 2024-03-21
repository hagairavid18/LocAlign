import os
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB import PDBList, MMCIFParser
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Atom import Atom

warnings.filterwarnings("ignore", category=PDBConstructionWarning)


class Protein:
    def __init__(self, pdb_name: str, chain_id: str, ligand_name: str, ligand_id: str, model_idx: int = 0 ) -> None:
  
        self._pdb_name = pdb_name
        self._chain_id = chain_id
        self._model_idx = model_idx
        self._ligand_name = ligand_name
        self._ligand_id = ligand_id
        self._structure: Structure = self._init_structure()
        self._save_ligand_model()
    
    def _init_structure(self) -> Structure:

        pdb_list = PDBList()
        pdb_file_path = pdb_list.retrieve_pdb_file(self._pdb_name, pdir=f'alligned_structures/{self._ligand_name}', file_format='mmCif')
        mmcif_parser = MMCIFParser()
        structure: Structure = mmcif_parser.get_structure(self._pdb_name, pdb_file_path)
        return structure
  
    def _save_ligand_model(self) -> None:
        peptide_model = self.get_model(self._model_idx)
        ligand_model = Protein.create_ligand_model(peptide_model, self._ligand_id, self._chain_id)
        save_dir = f'alligned_structures/{self._ligand_name}'
        io = PDBIO()
        io.set_structure(ligand_model)
        io.save(os.path.join(save_dir, f"{self._pdb_name}_ligand.pdb"))
    
    def get_model(self, model_idx: int, only_chain: bool = False) -> Model | Chain:
        if only_chain:
            return self._structure[model_idx][self._chain_id]
        
        return self._structure[model_idx]
    
    def get_ligand_atoms(self, ligand_id_name: int = 0) -> list[Atom]:
        return self._get_atoms(ligand_id_name)
    
    staticmethod
    def create_ligand_model(model: Model, ligand_id_name: str, chain_idx: int) -> Model:
        ligand_model = Model(model.id)

        chain: Chain = model[chain_idx]
        ligand_chain = Chain(chain.id)
        for residue in list(chain):
            if residue.id[0] == ligand_id_name:
                ligand_chain.add(residue.copy())
     
        ligand_model.add(ligand_chain)
        return ligand_model
    
    def _get_atoms(self, id: str = " ") -> list[Atom]:
        atoms =  [] 

        for residue in list(self.get_model(self._model_idx, True)):
            if residue.id[0] == id:
                atoms += list(residue)
     
        return atoms
    


  





