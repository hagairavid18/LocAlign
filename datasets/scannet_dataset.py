import logging
import os
import torch
import torch.nn.functional as F
from datasets import BasePairDataset
import pickle

from Bio.PDB import PDBParser
from Bio.PDB.Atom import PDBConstructionWarning
from Bio.PDB.Chain import Chain
from Bio.PDB.Structure import Structure
import warnings

from objects.protein import Protein
from utils.constants import LIGAND_DIR
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

logger = logging.getLogger(__name__)


class ScannetDataset(BasePairDataset):
    MAX_SEQUENCE_LENGTH = 1000

    def __init__(self, df_path: str, base_data_path: str = LIGAND_DIR, n_samples: int | None = None, min_cath: int = 0, seed: int| None = None) -> None:
        super().__init__(df_path, base_data_path, n_samples, min_cath, seed)
        self._mmcif_parser = PDBParser()

    def __getitem__(self, idx: int) -> dict[torch.Tensor]:
        row = self._df.iloc[idx]
        if row['n_transformations'] != 1: # TODO
            idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(idx)
        try:
            tar_embedding, tar_coordinates = self._read_embedding(ligand_id=row['Ligand_ID'], chain=row['ref_protein'])
            src_embedding, src_coordinates = self._read_embedding(ligand_id=row['Ligand_ID'], chain=row['mov_protein'])
        except Exception as e:
            idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(idx)
        try:
            src_pocket = self._read_pocket_coordinates(ligand_id=row['Ligand_ID'], p_name=row['mov_protein'])
        except Exception as e:
            idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(idx)

        tar_length, src_length = tar_embedding.shape[0], src_coordinates.shape[0]
        ret = {}
        ret['tar_embedding'] = F.pad(tar_embedding, (0, 0, 0, self.MAX_SEQUENCE_LENGTH - tar_length) )
        ret['tar_coordinates'] = F.pad(tar_coordinates, (0, 0, 0, self.MAX_SEQUENCE_LENGTH - tar_length) )
        ret['tar_mask'] = F.pad(torch.ones(tar_length), (0, self.MAX_SEQUENCE_LENGTH - tar_length), value=0).bool()
        ret['src_embedding'] = F.pad(src_embedding, (0, 0, 0, self.MAX_SEQUENCE_LENGTH - src_length) )
        ret['src_coordinates'] = F.pad(src_coordinates, (0, 0, 0, self.MAX_SEQUENCE_LENGTH - src_length))
        ret['src_mask'] = F.pad(torch.ones(src_length), (0, self.MAX_SEQUENCE_LENGTH - src_length), value=0).bool()
        ret['gt_R'] = torch.Tensor(row.to_dict()['rotations'][0][0])
        ret['gt_t'] = torch.Tensor(row.to_dict()['translations'][0][0])
        ret['max_length'] = max(tar_length, src_length)
        ret['metadata'] = row.to_dict()
        ret['src_pocket'] = F.pad(src_pocket, (0, 0, 0, self.MAX_SEQUENCE_LENGTH - src_pocket.shape[0]))
        ret['src_pocket_mask'] = F.pad(torch.ones(src_pocket.shape[0]), (0, self.MAX_SEQUENCE_LENGTH - src_pocket.shape[0]), value=0).bool()
        
        return ret
    
    def _read_embedding(self, ligand_id: str, chain: str) -> tuple[torch.Tensor, torch.Tensor]:
        embedding_path_tar = os.path.join(self._base_data_path, ligand_id,  chain + '_scannet.pkl')
        if not os.path.exists(embedding_path_tar):
            logger.info(f"Can't find embedding path for ligand: {ligand_id} protein: {chain}")
            raise ValueError
        with open(embedding_path_tar, 'rb') as f:
            data = pickle.load(f)
        embedding_ids, embeddings = list(data.keys()), list(data.values())
        embeddings = torch.stack([torch.tensor(arr) for arr in embeddings])
        
        non_ligand_model_path = os.path.join(self._base_data_path, ligand_id,  chain + '_non_ligand.ent')
        if not os.path.exists(non_ligand_model_path):
            # print(f"Can't find non ligand path for {chain}")
            raise ValueError
        structure: Structure = self._mmcif_parser.get_structure(chain, non_ligand_model_path)
        coordinates, residues_ids = [], []
        chain : Chain = list(list(structure)[0])[0]
        for residue in chain:
            for atom in residue:
                if atom.id == 'CA':
                    residues_ids.append((0, chain.id, residue.id[1]))
                    coordinates.append(torch.Tensor(atom.get_coord()))

        coordinates = torch.stack(coordinates)
        if not embedding_ids == residues_ids:
            intersection = set(embedding_ids) & set(residues_ids)

            embeddings = embeddings[[i for i, x in enumerate(embedding_ids) if x in intersection]]
            coordinates = coordinates[[i for i, x in enumerate(residues_ids) if x in intersection]]
            assert len(embeddings) == len(coordinates)
            # print("found diffs between embedding and resildues ids")
            # raise ValueError
        return embeddings, coordinates
    
    def _read_pocket_coordinates(self, ligand_id: str, p_name: str) -> torch.Tensor:
        structure: Structure = self._mmcif_parser.get_structure(p_name, f'{LIGAND_DIR}/{ligand_id}/{p_name}_pocket.pdb')
        pocket_coord, _ = Protein.get_residue_data(structure[0]['A'])
        pocket_coord = torch.tensor(pocket_coord)
        return pocket_coord
  
   
if __name__ == "__main__":
     # Example data
    from torch.utils.data import DataLoader

    data_path = '/home/iscb/wolfson/hagairavid/ligand_aligner/baseline_results/2024-07-11_10-55-38_57.csv'
    data_path = '/home/iscb/wolfson/hagairavid/ligand_aligner/baseline_results/2024-07-17_16-01-08_3000.csv'
    base_data_path = LIGAND_DIR


    # Create dataset and DataLoader
    dataset = ScannetDataset(data_path, base_data_path)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Iterate through the DataLoader
    print("Testing DataLoader:")
    for batch_idx, (samples, labels, infos) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print("Samples:", samples)
        print("Labels:", labels)
        print("Infos:", infos)
        print()