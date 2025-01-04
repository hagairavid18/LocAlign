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
from Bio.PDB.Polypeptide import protein_letters_3to1

import warnings

from utils.constants import LIGAND_DIR
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

logger = logging.getLogger(__name__)


class ScannetDataset(BasePairDataset):
    MAX_SEQUENCE_LENGTH = 1000
    MAX_ATOMS = 3000

    def __init__(self, df_path: str, base_data_path: str = LIGAND_DIR, infer_baseline: bool = False, n_samples: int | None = None, min_cath: int = 0, seed: int| None = None) -> None:
        super().__init__(df_path, base_data_path, n_samples, min_cath, seed)
        self._mmcif_parser = PDBParser()
        original_num_pairs = len(self._df)        
        self._infer_baseline = infer_baseline      
        self._df = self._df[self._df['has_scannet_embedding'] == True]  
        num_lost_pairs = original_num_pairs - len(self._df)
        print(f"Number of pairs lost due to missing embeddings: {num_lost_pairs}")

    def __getitem__(self, idx: int) -> dict[torch.Tensor]:
        row = self._df.iloc[idx]
        if self._infer_baseline:
            ret = {'metadata': row.to_dict()}
            ret['gt_R'] = torch.Tensor(row.to_dict()['rotations'][0][0])
            ret['gt_t'] = torch.Tensor(row.to_dict()['translations'][0][0])
            return ret

        try:
            tar_embedding, tar_coordinates, tar_res_indices, tar_all_coordinates, tar_residue_frames = self._read_embedding(ligand_id=row['Ligand_ID'], chain=row['ref_protein'])
            src_embedding, src_coordinates, src_res_indices, src_all_coordinates, src_residue_frames = self._read_embedding(ligand_id=row['Ligand_ID'], chain=row['mov_protein'])
            src_ligand_coordinates = self._read_ligand(ligand_id=row['Ligand_ID'], chain=row['mov_protein'])
        except Exception as e:
            print(f"Error reading embeddings for {row['Ligand_ID']} {row['mov_protein']} {row['ref_protein']}: {e}")
            idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(idx)

        try:
            src_pocket, src_pocket_residue_indices = self._read_pocket_coordinates(
                ligand_id=row['Ligand_ID'], 
                p_name=row['mov_protein']
            )
            tar_pocket, tar_pocket_residue_indices = self._read_pocket_coordinates(
                ligand_id=row['Ligand_ID'], 
                p_name=row['ref_protein']
            )

            filtered_src_pocket_residue_indices = torch.nonzero(torch.isin(src_res_indices, src_pocket_residue_indices )).squeeze()
            filtered_tar_pocket_residue_indices = torch.nonzero(torch.isin(tar_res_indices, tar_pocket_residue_indices )).squeeze()


        except Exception as e:
            idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(idx)

        tar_length, src_length = tar_embedding.shape[0], src_coordinates.shape[0]
        tar_pocket_length, src_pocket_length = filtered_tar_pocket_residue_indices.shape[0], filtered_src_pocket_residue_indices.shape[0]
        ret = {}
        # ret['tar_embedding'] = F.pad(tar_embedding, (0, 0, 0, self.MAX_SEQUENCE_LENGTH - tar_length))
        # ret['tar_coordinates'] = F.pad(tar_coordinates, (0, 0, 0, self.MAX_SEQUENCE_LENGTH - tar_length))
        # if ret['tar_coordinates'].shape[0] < self.MAX_SEQUENCE_LENGTH:
        #     print(f"tar_coordinates shape {ret['tar_coordinates'].shape}")
        # ret['tar_mask'] = F.pad(torch.ones(tar_length), (0, self.MAX_SEQUENCE_LENGTH - tar_length), value=0).bool()
        ret['tar_embedding'] = F.pad(tar_embedding[filtered_tar_pocket_residue_indices], (0, 0, 0, self.MAX_SEQUENCE_LENGTH - tar_pocket_length))
        ret['tar_coordinates'] = F.pad(tar_coordinates[filtered_tar_pocket_residue_indices], (0, 0, 0, self.MAX_SEQUENCE_LENGTH - tar_pocket_length))
        ret['tar_frames'] = F.pad(tar_residue_frames[filtered_tar_pocket_residue_indices], (0, 0, 0, 0, 0, self.MAX_SEQUENCE_LENGTH - tar_pocket_length))
        ret['tar_mask'] = F.pad(torch.ones(tar_pocket_length), (0, self.MAX_SEQUENCE_LENGTH - tar_pocket_length), value=0).bool()

        # ret['tar_all_coordinates'] = F.pad(tar_all_coordinates, (0, 0, 0, self.MAX_ATOMS - tar_all_coordinates.shape[0]))
        # ret['tar_all_mask'] = F.pad(torch.ones(tar_all_coordinates.shape[0]), (0, self.MAX_ATOMS - tar_all_coordinates.shape[0]), value=0).bool()
        
        # ret['src_embedding'] = F.pad(src_embedding, (0, 0, 0, self.MAX_SEQUENCE_LENGTH - src_length))
        # ret['src_coordinates'] = F.pad(src_coordinates, (0, 0, 0, self.MAX_SEQUENCE_LENGTH - src_length))
        # ret['src_mask'] = F.pad(torch.ones(src_length), (0, self.MAX_SEQUENCE_LENGTH - src_length), value=0).bool()
        ret['src_embedding'] = F.pad(src_embedding[filtered_src_pocket_residue_indices], (0, 0, 0, self.MAX_SEQUENCE_LENGTH - src_pocket_length))
        ret['src_coordinates'] = F.pad(src_coordinates[filtered_src_pocket_residue_indices], (0, 0, 0, self.MAX_SEQUENCE_LENGTH - src_pocket_length))
        ret['src_frames'] = F.pad(src_residue_frames[filtered_src_pocket_residue_indices], (0, 0, 0, 0, 0, self.MAX_SEQUENCE_LENGTH - src_pocket_length))
        ret['src_mask'] = F.pad(torch.ones(src_pocket_length), (0, self.MAX_SEQUENCE_LENGTH - src_pocket_length), value=0).bool()
        
        # ret['src_all_coordinates'] = F.pad(src_all_coordinates, (0, 0, 0, self.MAX_ATOMS - src_all_coordinates.shape[0]))
        # ret['src_all_mask'] = F.pad(torch.ones(src_all_coordinates.shape[0]), (0, self.MAX_ATOMS - src_all_coordinates.shape[0]), value=0).bool()
        
        ret['gt_R'] = torch.Tensor(row.to_dict()['rotations'][0][0])
        ret['gt_t'] = torch.Tensor(row.to_dict()['translations'][0][0])
        ret['max_length'] = max(tar_length, src_length)
        ret['metadata'] = row.to_dict()
        # ret['sample_weight'] = torch.tensor(2 * row['bbr'][0][0] + row['coverage'][0][0] + (1 - row['rmse'][0][0])) / 4
        ret['sample_weight'] = torch.tensor(row['coverage'][0][0])
        ret['src_pocket'] = F.pad(src_pocket, (0, 0, 0, self.MAX_SEQUENCE_LENGTH - src_pocket.shape[0]))
        ret['tar_pocket'] = F.pad(tar_pocket, (0, 0, 0, self.MAX_SEQUENCE_LENGTH - tar_pocket.shape[0]))
        ret['src_pocket_mask'] = F.pad(torch.ones(filtered_src_pocket_residue_indices.shape[0]), (0, self.MAX_SEQUENCE_LENGTH - filtered_src_pocket_residue_indices.shape[0]), value=0).bool()
        ret['tar_pocket_mask'] = F.pad(torch.ones(filtered_tar_pocket_residue_indices.shape[0]), (0, self.MAX_SEQUENCE_LENGTH - filtered_tar_pocket_residue_indices.shape[0]), value=0).bool()
        # ret['src_pocket_residue_indices'] = F.pad(filtered_src_pocket_residue_indices, (0, self.MAX_SEQUENCE_LENGTH - len(filtered_src_pocket_residue_indices)), value=0)  # Use -1 for padding residue indices
        # ret['tar_pocket_residue_indices'] = F.pad(filtered_tar_pocket_residue_indices, (0, self.MAX_SEQUENCE_LENGTH - len(filtered_tar_pocket_residue_indices)), value=0)  # Use -1 for padding residue indices
        # ret['src_residue_indices'] = F.pad(src_res_indices, (0, self.MAX_SEQUENCE_LENGTH - len(src_res_indices)), value=0)  # Use -1 for padding residue indices
        # ret['tar_residue_indices'] = F.pad(tar_res_indices, (0, self.MAX_SEQUENCE_LENGTH - len(tar_res_indices)), value=0)  # Use -1 for padding residue indices
        ret['src_ligand_coordinates'] = F.pad(src_ligand_coordinates, (0, 0, 0, self.MAX_SEQUENCE_LENGTH - len(src_ligand_coordinates)))
        ret['src_ligand_mask'] = F.pad(torch.ones(len(src_ligand_coordinates)), (0, self.MAX_SEQUENCE_LENGTH - len(src_ligand_coordinates)), value=0).bool()
        
        return ret
    
    def _read_embedding(self, ligand_id: str, chain: str) -> tuple[torch.Tensor, torch.Tensor]:
        # embedding_path_tar = os.path.join(self._base_data_path, ligand_id,  chain + '_scannet.pkl')
        embedding_path_tar = os.path.join(self._base_embedding_path, ligand_id,  chain + '_scannet_atoms.pkl')
        if not os.path.exists(embedding_path_tar):
            logger.info(f"Can't find embedding path for ligand: {ligand_id} protein: {chain}")
            raise ValueError
        with open(embedding_path_tar, 'rb') as f:
            data = pickle.load(f)
        embeddings = data.get("residue_embeddings", None)
        # atom_embeddings = data.get.get("atomic_embeddings", None)
        residue_frames = data.get("frames", None)
        # sequence_indices_atom = data_dict.get("sequence_indices_atom", None)
        embedding_ids = data.get("residue_ids", None)
        # sequence_indices_residue = data_dict.get("sequence_indices_residue", None)
        
        non_ligand_model_path = os.path.join(self._base_data_path, ligand_id,  chain + '_non_ligand.ent')
        if not os.path.exists(non_ligand_model_path):
            # print(f"Can't find non ligand path for {chain}")
            raise ValueError
        structure: Structure = self._mmcif_parser.get_structure(chain, non_ligand_model_path)
        ca_coordinates, all_coordinates, residues_ids = [], [], []
        corrupt_residues_idx = []
        chain: Chain = list(list(structure)[0])[0]
        residue_lookup = {residue.id[1]: residue for residue in chain}
        for embedding_id in embedding_ids:
            residue = residue_lookup.get(int(embedding_id[2]))
            # if not residue.id[1] > -1: #TODO: check if this is the correct way to filter out non amino acids
            #     continue
            n_atoms = 0
            if "CA" not in residue.child_dict:
                corrupt_residues_idx.append(int(embedding_id[2]))
                # print(f"CA atom not found in residue {residue.id[1]} in chain {chain}")
                # raise ValueError("CA atom not found")
            for atom in residue:
                if atom.id == "H":
                    print(f"found H atom {chain}")
                    continue
                
                if atom.id == 'CA':
                    residues_ids.append((0, chain.id, residue.id[1]))
                    ca_coordinates.append(torch.Tensor(atom.get_coord()))
                    # all_coordinates.append(torch.cat((torch.tensor(atom.get_coord()), torch.tensor([residue.id[1]])), dim=0))
                if atom.id != 'CA' and n_atoms < 2:
                    # all_coordinates.append(torch.cat((torch.tensor(atom.get_coord()), torch.tensor([residue.id[1]])), dim=0))
                    n_atoms += 1
                # if n_atoms < 2:
                #     print(f"Adding dummy atoms to residue {residue.id[1]} in chain {chain}")
                #     dummy_coord = torch.randn(3) * 1e6  # Dummy coordinates (could be zero or other values)
                #     dummy_residue_idx = -1  # Dummy residue index
                #     # Add dummy atom to fill the missing atom
                #     for _ in range(2 - n_atoms):  # Add the necessary number of dummy atoms
                #         all_coordinates.append(torch.cat((dummy_coord, torch.tensor([dummy_residue_idx])), dim=0))
        ca_coordinates = torch.stack(ca_coordinates)
        if len(corrupt_residues_idx) > 0:
            valid_residue_list_indices = [i for i, tup in enumerate(embedding_ids) if int(tup[2]) not in corrupt_residues_idx]
            embeddings = [embeddings[i] for i in valid_residue_list_indices]
            embedding_ids = [embedding_ids[i] for i in valid_residue_list_indices]
            residue_frames = [residue_frames[i] for i in valid_residue_list_indices]
        # all_coordinates = torch.stack(all_coordinates)
        residue_indices = [int(tup[2]) for tup in embedding_ids]
        assert all([len(embeddings) == len(ca_coordinates), len(embeddings) == len(residue_indices)])
        return torch.Tensor(embeddings), ca_coordinates, torch.Tensor(residue_indices), all_coordinates, torch.Tensor(residue_frames)
    
    def _read_ligand(self, ligand_id: str, chain: str) -> tuple[torch.Tensor, torch.Tensor]:
        ligand_model_path = os.path.join(self._base_data_path, ligand_id,  chain + '_ligand.pdb')
        if not os.path.exists(ligand_model_path):
            print(f"Can't find ligand path for {chain}")
            raise ValueError
        structure: Structure = self._mmcif_parser.get_structure(chain, ligand_model_path)
        coordinates, residues_ids = [], []
        chain : Chain = list(list(structure)[0])[0]
        if len(list(chain)) > 1: 
            print(f"ligand {ligand_id} found in {chain} more than once")
        for residue in chain:
            for atom in residue:
                coordinates.append(torch.Tensor(atom.get_coord()))

        coordinates = torch.stack(coordinates)

        return coordinates

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
            if not isinstance(data, dict) or 'pocket_ca_coords' not in data or 'residue_indices' not in data:
                raise ValueError("Invalid pickle format. Expected a dictionary with 'pocket_coords' and 'residue_indices' keys.")

            # Extract coordinates and residue indices
            pocket_coords = torch.tensor(data['pocket_ca_coords'], dtype=torch.float32)
            residue_indices = torch.tensor(data['residue_indices'], dtype=torch.int64)

            if pocket_coords.numel() == 0:
                raise ValueError(f"src_pocket is empty for {p_name}. Ensure the dataset entry is valid.")

            return pocket_coords, residue_indices

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