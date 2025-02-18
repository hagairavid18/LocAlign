import logging
import os
import torch
import torch.nn.functional as F
from datasets import BasePairDataset
import pickle

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Atom import PDBConstructionWarning
from Bio.PDB.Chain import Chain
from Bio.PDB.Structure import Structure

import warnings

from utils.constants import LIGAND_DIR
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

logger = logging.getLogger(__name__)


class ScannetDataset(BasePairDataset):
    MAX_LENGTH_DICT = {'residue': 1000, 'atom': 2500, 'pocket': 1000}

    def __init__(self, df_path: str, base_data_path: str = LIGAND_DIR, infer_baseline: bool = False, level: str = 'residue', n_samples: int | None = None, min_cath: int = 0, max_cath: int = 8, bbr_filter_ratio: float = 0.0, seed: int| None = None) -> None:
        super().__init__(df_path, base_data_path, n_samples, min_cath, max_cath, seed=seed, bbr_filter_ratio=bbr_filter_ratio)
        self._mmcif_parser = PDBParser()
        original_num_pairs = len(self._df)        
        self._infer_baseline = infer_baseline      
        num_lost_pairs = original_num_pairs - len(self._df)
        assert level in ['residue', 'atom', 'pocket'], "level must be one of ['residue', 'atom', 'pocket']"
        self._level = level
        print(f"Number of pairs lost due to missing embeddings: {num_lost_pairs}")

    def __getitem__(self, idx: int) -> dict[torch.Tensor]:
        row = self._df.iloc[idx]
        if row['mov_protein'] == '3n6r' or row['mov_protein'] == '4hyj' or idx in [5094, 5095]:
                print(f"idx {idx} {row['Ligand_ID']} {row['mov_protein']} {row['ref_protein']}")
                idx = torch.randint(0, len(self), (1,)).item()
                return self.__getitem__(idx)

        # if self._infer_baseline:
        #     return {
        #         'metadata': row.to_dict(),
        #         'gt_R': torch.Tensor(row['rotations'][0][0]),
        #         'gt_t': torch.Tensor(row['translations'][0][0]),
        #     }

        try:
            embedding_dicts = {
                "tar": self._read_embedding(ligand_id=row['Ligand_ID'], chain=row['ref_protein']),
                "src": self._read_embedding(ligand_id=row['Ligand_ID'], chain=row['mov_protein'])
            }
            src_ligand_coordinates = self._read_ligand(ligand_id=row['Ligand_ID'], chain=row['mov_protein'])
            tar_ligand_coordinates = self._read_ligand(ligand_id=row['Ligand_ID'], chain=row['ref_protein'])
        except Exception as e:
            print(f"Error reading embeddings for {row['Ligand_ID']} {row['mov_protein']} {row['ref_protein']}: {e}")
            idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(idx)

        try:
            pocket_data = {
                "src": self._read_pocket_coordinates(ligand_id=row['Ligand_ID'], p_name=row['mov_protein']),
                "tar": self._read_pocket_coordinates(ligand_id=row['Ligand_ID'], p_name=row['ref_protein'])
            }
            
        except Exception as e:
            print(f"Error reading pocket data: {e}")
            idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(idx)

        ret = {}
        for key in ["src", "tar"]:
            embedding_dict = embedding_dicts[key]
            pocket_residue_indices = pocket_data[key]

            indices_for_pocket = torch.isin(embedding_dict["sequence_indices_atom"], pocket_residue_indices)
            if indices_for_pocket.sum() < 10:
                print(f"Error: {indices_for_pocket.sum()} indices for pocket")
                idx = torch.randint(0, len(self), (1,)).item()
                return self.__getitem__(idx)

            embedding_dict['pocket_embeddings'] = embedding_dict['atom_embeddings'][indices_for_pocket]
            embedding_dict['pocket_frames'] = embedding_dict['atom_frames'][indices_for_pocket]
            pocket_length = embedding_dict['pocket_frames'].shape[0]
            length = embedding_dict[f'{self._level}_embeddings'].shape[0]

            ret[f'{key}_embedding'] = F.pad(embedding_dict[f'{self._level}_embeddings'], (0, 0, 0, self.MAX_LENGTH_DICT[self._level] - length))
            ret[f'{key}_frames'] = F.pad(embedding_dict[f'{self._level}_frames'], (0, 0, 0, 0, 0, self.MAX_LENGTH_DICT[self._level] - length))
            ret[f'{key}_mask'] = F.pad(torch.ones(length), (0, self.MAX_LENGTH_DICT[self._level] - length), value=0).bool()
          
            ret[f'{key}_pocket_embedding'] = F.pad(embedding_dict['pocket_embeddings'], (0, 0, 0, self.MAX_LENGTH_DICT['pocket'] - pocket_length))
            ret[f'{key}_pocket_frames'] = F.pad(embedding_dict['pocket_frames'], (0, 0, 0, 0, 0, self.MAX_LENGTH_DICT['pocket'] - pocket_length))
            ret[f'{key}_pocket_mask'] = F.pad(torch.ones(pocket_length), (0, self.MAX_LENGTH_DICT['pocket'] - pocket_length), value=0).bool()

        ret['max_length'] = max(embedding_dicts['src'][f'{self._level}_embeddings'].shape[0], embedding_dicts['tar'][f'{self._level}_embeddings'].shape[0])
        ret['gt_R'] = torch.Tensor(row['rotations'][0][0])
        ret['gt_t'] = torch.Tensor(row['translations'][0][0])
        
        ret['metadata'] = row.to_dict()
        ret['metadata']['idx'] = idx
        ret['sample_weight'] = torch.tensor(row['sample_weight'])
        ret['src_ligand_coordinates'] = F.pad(src_ligand_coordinates, (0, 0, 0, self.MAX_LENGTH_DICT['residue'] - len(src_ligand_coordinates)))
        ret['tar_ligand_coordinates'] = F.pad(tar_ligand_coordinates, (0, 0, 0, self.MAX_LENGTH_DICT['residue'] - len(tar_ligand_coordinates)))

        ret['src_ligand_mask'] = F.pad(torch.ones(len(src_ligand_coordinates)), (0, self.MAX_LENGTH_DICT['residue'] - len(src_ligand_coordinates)), value=0).bool()
        ret['tar_ligand_mask'] = F.pad(torch.ones(len(tar_ligand_coordinates)), (0, self.MAX_LENGTH_DICT['residue'] - len(tar_ligand_coordinates)), value=0).bool()
        for key, tensor in ret.items():
            if isinstance(tensor, torch.Tensor):
                if torch.isnan(tensor).any():  # Checks if there are any NaNs in the tensor
                    print(f"NaN detected in {key}")
                    idx = torch.randint(0, len(self), (1,)).item()
                    return self.__getitem__(idx)
        return ret

    def _read_embedding(self, ligand_id: str, chain: str) -> tuple[torch.Tensor, torch.Tensor]:
        embedding_path = os.path.join(self._base_embedding_path, ligand_id,  chain + '_scannet_atoms.pkl')
        if not os.path.exists(embedding_path):
            logger.info(f"Can't find embedding path for ligand: {ligand_id} protein: {chain}")
            raise ValueError
        with open(embedding_path, 'rb') as f:
            data = pickle.load(f)
        # Example data
        residue_embeddings = data["residue_embeddings"]
        residue_frames = data["residue_frames"]
        residue_ids = data["residue_ids"]
        atom_embeddings = data["atomic_embeddings"]
        atom_residue_index = data["sequence_indices_atom"]  # Residue index for each atom
        atom_frames = data["atomic_frames"]
        residue_embeddings_up_pooled = residue_embeddings[atom_residue_index]
        atomic_plus_residue_embedding = np.concatenate((atom_embeddings, residue_embeddings_up_pooled),axis=-1)
        
        if self._level == 'atom':
            atom_sampled_indices = np.random.choice(len(atom_embeddings), size=min(len(atom_embeddings), self.MAX_LENGTH_DICT['atom']), replace=False)
            atom_embeddings = atom_embeddings[atom_sampled_indices]
            atom_residue_index = atom_residue_index[atom_sampled_indices]
            atom_frames = atom_frames[atom_sampled_indices]
            atomic_plus_residue_embedding = atomic_plus_residue_embedding[atom_sampled_indices]

        residue_indices = residue_ids[:, -1].astype(int)  # Extract residue indices (last column of residue_ids)
        atom_residue_index = residue_indices[atom_residue_index]

        # Validate input data
        if residue_frames is None or residue_ids is None or atom_residue_index is None:
            raise ValueError("Missing required data: 'frames', 'residue_ids', or 'sequence_indices_atom'.")

        ret_dict = {
            'atom_frames': atom_frames,
            'residue_frames': residue_frames,
            'atom_embeddings': atomic_plus_residue_embedding,
            'residue_embeddings': residue_embeddings,
            'residue_indices': residue_indices,
            'sequence_indices_atom': atom_residue_index
        }
        return {key: torch.tensor(value) for key, value in ret_dict.items()}

    def _read_ligand(self, ligand_id: str, chain: str) -> tuple[torch.Tensor, torch.Tensor]:
        ligand_model_path = os.path.join(self._base_data_path, ligand_id,  chain + '_ligand.pdb')
        if not os.path.exists(ligand_model_path):
            print(f"Can't find ligand path for {chain}")
            raise ValueError
        structure: Structure = self._mmcif_parser.get_structure(chain, ligand_model_path)
        coordinates = []
        chain : Chain = list(list(structure)[0])[0]
        if len(list(chain)) > 1: 
            print(f"ligand {ligand_id} found in {chain} more than once")
        for residue in chain:
            for atom in residue:
                coordinates.append(torch.Tensor(atom.get_coord()))

        return torch.stack(coordinates)

    def _read_pocket_coordinates(self, ligand_id: str, p_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reads pocket CA coordinates and their residue indices from a single saved pickle file.

        Args:
            ligand_id (str): Identifier for the ligand.
            p_name (str): Protein name.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - pocket_coords: Tensor of pocket CA coordinates.
                - residue_indices: Tensor of corresponding residue indices.
        """
        # Define path to the pickle file
        pickle_path = f'{LIGAND_DIR}/{ligand_id}/{p_name}_pocket_data.pkl'

        try:
            # Load the data from the pickle file
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)

            # Ensure the loaded data contains the expected keys
            if not isinstance(data, dict)  or 'residue_indices' not in data:
                raise ValueError("Invalid pickle format. Expected a dictionary with 'pocket_coords' and 'residue_indices' keys.")

            # Extract coordinates and residue indices
            # pocket_coords = torch.tensor(data['pocket_ca_coords'], dtype=torch.float32)
            residue_indices = torch.tensor(data['residue_indices'], dtype=torch.int64)

            # if pocket_coords.numel() == 0:
            #     raise ValueError(f"src_pocket is empty for {p_name}. Ensure the dataset entry is valid.")

            return residue_indices

        except Exception as e:
            logging.error(f"Error loading pocket data from {pickle_path}: {e}")
            raise
  
   
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