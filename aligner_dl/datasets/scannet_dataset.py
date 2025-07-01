import warnings
import logging
import os
import torch
torch.set_float32_matmul_precision('medium')  # or 'high'
import torch.nn.functional as F
import pickle
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Atom import PDBConstructionWarning
from Bio.PDB.Chain import Chain
from Bio.PDB.Structure import Structure
import esm
from Bio.PDB import PDBParser

from datasets import BasePairDataset
from utils.constants import LIGAND_DIR

warnings.filterwarnings("ignore", category=PDBConstructionWarning)

logger = logging.getLogger(__name__)


class ScannetDataset(BasePairDataset):

    def __init__(self, df_path: str, base_embedding_path: str, base_data_path: str = LIGAND_DIR, infer_baseline: bool = False, level: str = 'residue', n_samples: int | None = None, min_cath: int = 0, max_cath: int = 8, bbc_filter_ratio: float = 0.0, seed: int| None = None, max_length: int = None, inference: bool = False, ligand_column: str = 'Ligand_ID') -> None:
        super().__init__(df_path, base_data_path, base_embedding_path, n_samples, min_cath, max_cath, seed=seed, bbc_filter_ratio=bbc_filter_ratio, inference=inference)
        self.MAX_LENGTH_DICT = {'residue': 1000, 'atom': 5000, 'pocket': 800}
        self._mmcif_parser = PDBParser()
        original_num_pairs = len(self._df)        
        self._infer_baseline = infer_baseline
        self.ligand_column = ligand_column
        self._esm_model, self._esm_alphabet = getattr(esm.pretrained, "esm2_t30_150M_UR50D")()
        self._batch_converter = self._esm_alphabet.get_batch_converter()
        self._esm_model = self._esm_model.eval()
        # if torch.cuda.is_available():
        #     self._esm_model = self._esm_model.cuda()
        num_lost_pairs = original_num_pairs - len(self._df)
        if max_length is not None:
            # self.MAX_LENGTH_DICT['residue'] = max_length
            self.MAX_LENGTH_DICT[level] = max_length
            # self.MAX_LENGTH_DICT['pocket'] = max_length
        assert level in ['residue', 'atom', 'pocket'], "level must be one of ['residue', 'atom', 'pocket']"
        self._level = level
        print(f"Number of pairs lost due to missing embeddings: {num_lost_pairs}")

    def __getitem__(self, idx: int) -> dict[torch.Tensor]:
        # ligand_column = 'Ligand_ID' if 'Ligand_ID' in self._df.columns else 'ligand_id'
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
            esm_embeddings_tar = self.extract_esm_embeddings_by_resseq(
                pdb_file=os.path.join(self._base_data_path, row[self.ligand_column], row['ref_protein'] + '_non_ligand.ent'),
                # residue_numbers=embedding_dicts['tar']['residue_residue_indices'].tolist(),
            )
            esm_embeddings_src = self.extract_esm_embeddings_by_resseq(
                pdb_file=os.path.join(self._base_data_path, row[self.ligand_column], row['mov_protein'] + '_non_ligand.ent'),
                # residue_numbers=embedding_dicts['src']['residue_residue_indices'].tolist(),
            )
            embedding_dicts = {
                "tar": self._read_embedding(ligand_id=row[self.ligand_column], chain=row['ref_protein'], esm_embedding_dict= esm_embeddings_tar),
                "src": self._read_embedding(ligand_id=row[self.ligand_column], chain=row['mov_protein'], esm_embedding_dict= esm_embeddings_src),
            }
            print(f"Read embeddings for {row[self.ligand_column]} {row['mov_protein']} {row['ref_protein']}")
            if not self.inference:
                src_ligand_coordinates, src_atom_ids = self._read_ligand(ligand_id=row[self.ligand_column], chain=row['mov_protein'])
                tar_ligand_coordinates, tar_atom_ids = self._read_ligand(ligand_id=row[self.ligand_column], chain=row['ref_protein'])
                if not src_atom_ids == tar_atom_ids:
                    shared_atom_ids = set(src_atom_ids).intersection(tar_atom_ids)
                    src_ligand_coordinates = src_ligand_coordinates[torch.tensor([src_atom_ids.index(atom_id) for atom_id in shared_atom_ids])]
                    tar_ligand_coordinates = tar_ligand_coordinates[torch.tensor([tar_atom_ids.index(atom_id) for atom_id in shared_atom_ids])]
                    assert src_ligand_coordinates.shape == tar_ligand_coordinates.shape

        except Exception as e:
            # print(f"Error reading embeddings for {row[ligand_column]} {row[ligand_column]} {row[ligand_column]}: {e}")
            idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(idx)
        
            # embedding_dicts['tar']['esm_embeddings'] = torch.stack([esm_embeddings[resseq] for resseq in row['residue_numbers']])
            # embedding_dicts['src']['esm_embeddings'] = torch.stack([esm_embeddings[resseq] for resseq in row['residue_numbers']])
      

        if not self.inference:
            try:
                pocket_data = {
                    "src": self._read_pocket_coordinates(ligand_id=row[self.ligand_column], p_name=row['mov_protein']),
                    "tar": self._read_pocket_coordinates(ligand_id=row[self.ligand_column], p_name=row['ref_protein'])
                }
                
            except Exception as e:
                print(f"Error reading pocket data: {e}")
                idx = torch.randint(0, len(self), (1,)).item()
                return self.__getitem__(idx)

        ret = {}
        for key in ["src", "tar"]:
            embedding_dict = embedding_dicts[key]


            length = embedding_dict[f'{self._level}_embeddings'].shape[0]

            ret[f'{key}_embedding'] = F.pad(embedding_dict[f'{self._level}_embeddings'], (0, 0, 0, self.MAX_LENGTH_DICT[self._level] - length))
            ret[f'{key}_frames'] = F.pad(embedding_dict[f'{self._level}_frames'], (0, 0, 0, 0, 0, self.MAX_LENGTH_DICT[self._level] - length))
            ret[f'{key}_residue_indices'] = F.pad(embedding_dict[f'{self._level}_residue_indices'], (0, self.MAX_LENGTH_DICT[self._level] - length))
            ret[f'{key}_mask'] = F.pad(torch.ones(length), (0, self.MAX_LENGTH_DICT[self._level] - length), value=0).bool()
            if self.inference:
                continue
          
            
            pocket_residue_indices = pocket_data[key]
            indices_for_pocket = torch.isin(embedding_dict["atom_residue_indices"], pocket_residue_indices)
            if indices_for_pocket.sum() < 10:
                print(f"Error: {indices_for_pocket.sum()} indices for pocket")
                idx = torch.randint(0, len(self), (1,)).item()
                return self.__getitem__(idx)
            embedding_dict['pocket_embeddings'] = embedding_dict['atom_embeddings'][indices_for_pocket]
            embedding_dict['pocket_frames'] = embedding_dict['atom_frames'][indices_for_pocket]
            pocket_length = embedding_dict['pocket_frames'].shape[0]
            ret[f'{key}_pocket_embedding'] = F.pad(embedding_dict['pocket_embeddings'], (0, 0, 0, self.MAX_LENGTH_DICT['pocket'] - pocket_length))
            ret[f'{key}_pocket_frames'] = F.pad(embedding_dict['pocket_frames'], (0, 0, 0, 0, 0, self.MAX_LENGTH_DICT['pocket'] - pocket_length))
            ret[f'{key}_pocket_mask'] = F.pad(torch.ones(pocket_length), (0, self.MAX_LENGTH_DICT['pocket'] - pocket_length), value=0).bool()

        ret['max_length'] = max(embedding_dicts['src'][f'{self._level}_embeddings'].shape[0], embedding_dicts['tar'][f'{self._level}_embeddings'].shape[0])
        ret['metadata'] = row.to_dict()
        ret['metadata']['idx'] = idx
        if self.inference:
            return ret
        
        ret['gt_R'] = torch.Tensor(row['rotations'][0][0])
        ret['gt_t'] = torch.Tensor(row['translations'][0][0])
        
        # ret['sample_weight'] = torch.tensor(row['sample_weight'])
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

    def _read_embedding(
        self, 
        chain: str, 
        ligand_id: str,
        esm_embedding_dict: dict[int, torch.Tensor] | None = None
        ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding_path = os.path.join(self._base_embedding_path, ligand_id,  chain + '_scannet_atoms.pkl')
        # embedding_path = os.path.join(self._base_embedding_path, ligand_id,  chain + '.pkl')
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
        

        residue_indices = residue_ids[:, -1].astype(int)  # Extract residue indices (last column of residue_ids)
        atom_residue_index = residue_indices[atom_residue_index]

        try:
            esm_vectors = np.stack([
                esm_embedding_dict[res_id] for res_id in residue_indices
            ])  # shape: [num_residues, 1280]
        except KeyError as e:
            raise ValueError(f"Missing ESM embedding for residue {e} in ligand {ligand_id}, chain {chain}")

        # === Map ESM embeddings to atoms ===
        esm_per_atom = esm_vectors[data["sequence_indices_atom"]]

        # === Concatenate ===
        atomic_plus_residue_embedding = np.concatenate([atomic_plus_residue_embedding, esm_per_atom], axis=-1)
        if self._level == 'atom':
            atom_sampled_indices = np.random.choice(len(atom_embeddings), size=min(len(atom_embeddings), self.MAX_LENGTH_DICT['atom']), replace=False)
            atom_embeddings = atom_embeddings[atom_sampled_indices]
            atom_residue_index = atom_residue_index[atom_sampled_indices]
            atom_frames = atom_frames[atom_sampled_indices]
            atomic_plus_residue_embedding = atomic_plus_residue_embedding[atom_sampled_indices]

        ret_dict = {
            'atom_frames': atom_frames,
            'residue_frames': residue_frames,
            'atom_embeddings': atomic_plus_residue_embedding,
            'residue_embeddings': residue_embeddings,
            'residue_residue_indices': residue_indices,
            'atom_residue_indices': atom_residue_index
        }
        return {key: torch.tensor(value) for key, value in ret_dict.items()}

    def _read_ligand(self, ligand_id: str, chain: str) -> tuple[torch.Tensor, list[str]]:
        ligand_model_path = os.path.join(self._base_data_path, ligand_id,  chain + '_ligand.pdb')
        if not os.path.exists(ligand_model_path):
            print(f"Can't find ligand path for {chain}")
            raise ValueError
        structure: Structure = self._mmcif_parser.get_structure(chain, ligand_model_path)
        coordinates, ids = [], []
        chain : Chain = list(list(structure)[0])[0]
        if len(list(chain)) > 1: 
            print(f"ligand {ligand_id} found in {chain} more than once")
        for residue in chain:
            for atom in residue:
                coordinates.append(torch.Tensor(atom.get_coord()))
                ids.append(atom.id)

        return torch.stack(coordinates), ids

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
        pickle_path = f'{self._base_data_path}/{ligand_id}/{p_name}_pocket_data.pkl'

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


    def extract_esm_embeddings_by_resseq(self, pdb_file: str) -> dict:
        """
        Extract ESM embeddings for all standard residues in a PDB file, with caching.
        
        Returns:
            Dict mapping resseq (int) → ESM embedding tensor of shape [1280]
        """
        import os
        import hashlib
        from Bio.PDB import PDBParser, PPBuilder

        # === Caching ===
        cache_dir = os.path.join(self._base_data_path, "esm_cache_esm2_t30_150M_UR50D")
        os.makedirs(cache_dir, exist_ok=True)

        # Use a hash of the pdb path for unique filename
        pdb_hash = hashlib.md5(pdb_file.encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"{pdb_hash}.pt")

        if os.path.exists(cache_path):
            # print(f"Loading cached ESM embeddings from {cache_path}")
            return torch.load(cache_path)

        # === Compute ESM embeddings ===
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("pdb", pdb_file)
        ppb = PPBuilder()

        model = list(structure)[0]
        chain = list(model)[0]
        peptides = ppb.build_peptides(chain)

        if not peptides:
            raise ValueError("No peptide chains found")

        sequence = ""
        residues = []

        for peptide in peptides:
            sequence += str(peptide.get_sequence())
            residues.extend(peptide)

        # Map resseq → index in sequence
        resseq_to_index = {
            res.get_id()[1]: idx for idx, res in enumerate(residues)
            if res.get_id()[0] == ' '
        }

        data = [("sequence", sequence)]
        _, _, tokens = self._batch_converter(data)

        with torch.no_grad():
            num_layers = self._esm_model.num_layers
            results = self._esm_model(tokens, repr_layers=[num_layers], return_contacts=False)

        layer = list(results["representations"].keys())[-1]
        reps = results["representations"][layer][0, 1:len(sequence)+1]

        embedding_dict = {
            resseq: reps[idx].cpu()  # Save as CPU tensors for portability
            for resseq, idx in resseq_to_index.items()
        }

        # === Save to cache ===
        torch.save(embedding_dict, cache_path)

        return embedding_dict



   
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