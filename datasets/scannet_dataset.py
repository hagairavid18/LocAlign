import logging
import os
import torch
from datasets import BasePairDataset
import pickle
from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
from Bio.PDB import PDBParser

logger = logging.getLogger(__name__)


class ScannetDataset(BasePairDataset):
    def __init__(self, df_path: str, base_data_path: str, n_samples: int):
        super().__init__(df_path, base_data_path, n_samples)
        self._mmcif_parser = PDBParser()

    def __getitem__(self, idx: int) -> dict[torch.Tensor]:
        row = self._df.iloc[idx]
        if row['n_transformations'] != 1: # TODO
            idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(idx)
        try:
            ref_embedding, ref_coordinates = self._read_embedding(ligand_id=row['Ligand_ID'], chain=row['ref_protein'])
            mov_embedding, mov_coordinates = self._read_embedding(ligand_id=row['Ligand_ID'], chain=row['mov_protein'])
        except Exception as e:
            idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(idx)

        ret = {}
        ret['ref_embedding'] = ref_embedding
        ret['ref_coordinates'] = ref_coordinates
        ret['mov_embedding'] = mov_embedding
        ret['mov_coordinates'] = mov_coordinates
        ret['metadata'] = row.to_dict()
        ret['row_idx'] = idx
        
        return ret
    
    def _read_embedding(self, ligand_id: str, chain: str):
        embedding_path_ref = os.path.join(self._base_data_path, ligand_id,  chain + '_scannet.pkl')
        if not os.path.exists(embedding_path_ref):
            logger.info(f"Can't find embedding path for ligand: {ligand_id} protein: {chain}")
            raise ValueError
        with open(embedding_path_ref, 'rb') as f:
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
    
  
   
if __name__ == "__main__":
     # Example data
    from torch.utils.data import Dataset, DataLoader

    data_path = '/home/iscb/wolfson/hagairavid/ligand_aligner/baseline_results/2024-07-11_10-55-38_57.csv'
    data_path = '/home/iscb/wolfson/hagairavid/ligand_aligner/baseline_results/2024-07-17_16-01-08_3000.csv'
    base_data_path = os.path.join('/home/iscb/wolfson/hagairavid/ligand_aligner/ligands')


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