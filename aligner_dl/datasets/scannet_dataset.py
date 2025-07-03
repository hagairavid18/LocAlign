import warnings
import logging
import os
import torch
import torch.nn.functional as F
import pickle
import numpy as np
from Bio.PDB.Atom import PDBConstructionWarning
from Bio.PDB.Chain import Chain
from Bio.PDB.Structure import Structure
import esm
import hashlib
from Bio.PDB import PDBParser, PPBuilder

from datasets import BasePairDataset
from utils.constants import LIGAND_DIR

torch.set_float32_matmul_precision('medium')  # or 'high'
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

logger = logging.getLogger(__name__)


class ScannetDataset(BasePairDataset):

    def __init__(
            self, 
            df_path: str, 
            base_embedding_path: str, 
            base_data_path: str = LIGAND_DIR, 
            infer_baseline: bool = False, 
            level: str = 'residue', 
            n_samples: int | None = None, 
            min_cath: int = 0, 
            max_cath: int = 8, 
            bbc_filter_ratio: float = 0.0, 
            seed: int| None = None, 
            max_length: int = None, 
            inference: bool = False, 
            ligand_column: str = 'Ligand_ID',
            esm_model: str = None,
            ) -> None:
        """
        Initializes the ScannetDataset.

        Args:
            df_path (str): _path to the CSV file containing the dataset metadata.
            base_embedding_path (str): 
            base_data_path (str, optional): _description_. Defaults to LIGAND_DIR.
            infer_baseline (bool, optional): _description_. Defaults to False.
            level (str, optional): _description_. Defaults to 'residue'.
            n_samples (int | None, optional): _description_. Defaults to None.
            min_cath (int, optional): _description_. Defaults to 0.
            max_cath (int, optional): _description_. Defaults to 8.
            bbc_filter_ratio (float, optional): _description_. Defaults to 0.0.
            seed (int | None, optional): _description_. Defaults to None.
            max_length (int, optional): _description_. Defaults to None.
            inference (bool, optional): _description_. Defaults to False.
            ligand_column (str, optional): _description_. Defaults to 'Ligand_ID'.
        """        
        super().__init__(
            df_path, 
            base_data_path, 
            base_embedding_path, 
            n_samples,
            ligand_column=ligand_column, 
            min_cath=min_cath, 
            max_cath=max_cath, 
            seed=seed, 
            bbc_filter_ratio=bbc_filter_ratio, 
            inference=inference)
        
        self.MAX_LENGTH_DICT = {'residue': 1000, 'atom': 5000, 'pocket': 800}
        self._mmcif_parser = PDBParser()
        
        self._infer_baseline = infer_baseline
        
        if max_length is not None:
            # self.MAX_LENGTH_DICT['residue'] = max_length
            self.MAX_LENGTH_DICT[level] = max_length
            # self.MAX_LENGTH_DICT['pocket'] = max_length
        assert level in ['residue', 'atom', 'pocket'], "level must be one of ['residue', 'atom', 'pocket']"
        self._level = level

        if esm_model is not None:
            self._init_esm_model(esm_model)

    def _init_esm_model(self, esm_model: str) -> None:
        self._esm_model, self._esm_alphabet = getattr(esm.pretrained, esm_model)()
        self._batch_converter = self._esm_alphabet.get_batch_converter()
        self._esm_model = self._esm_model.eval()
        self._ppb_builder = PPBuilder()
        self._cache_dir = os.path.join(self._base_data_path, f"esm_cache_{esm_model}")
        os.makedirs(self._cache_dir, exist_ok=True)

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
            esm_embeddings_tar = self.extract_esm_embeddings(
                pdb_file=os.path.join(self._base_data_path, row[self._ligand_column], row['ref_protein'] + '_non_ligand.ent'),
                chain_name=row['ref_protein'] + '_' + row['ref_chain']
            )
            esm_embeddings_src = self.extract_esm_embeddings(
                pdb_file=os.path.join(self._base_data_path, row[self._ligand_column], row['mov_protein'] + '_non_ligand.ent'),
                chain_name=row['mov_protein'] + '_' + row['mov_chain']
            ) 
            embedding_dicts = {
                "tar": self._read_embedding(ligand_id=row[self._ligand_column], chain=row['ref_protein'], esm_embedding_dict= esm_embeddings_tar),
                "src": self._read_embedding(ligand_id=row[self._ligand_column], chain=row['mov_protein'], esm_embedding_dict= esm_embeddings_src),
            }
            print(f"Read embeddings for {row[self._ligand_column]} {row['mov_protein']} {row['ref_protein']}")
            if not self.inference:
                src_ligand_coordinates, src_atom_ids = self._read_ligand(ligand_id=row[self._ligand_column], chain=row['mov_protein'])
                tar_ligand_coordinates, tar_atom_ids = self._read_ligand(ligand_id=row[self._ligand_column], chain=row['ref_protein'])
                if not src_atom_ids == tar_atom_ids:
                    shared_atom_ids = set(src_atom_ids).intersection(tar_atom_ids)
                    src_ligand_coordinates = src_ligand_coordinates[torch.tensor([src_atom_ids.index(atom_id) for atom_id in shared_atom_ids])]
                    tar_ligand_coordinates = tar_ligand_coordinates[torch.tensor([tar_atom_ids.index(atom_id) for atom_id in shared_atom_ids])]
                    assert src_ligand_coordinates.shape == tar_ligand_coordinates.shape

        except Exception as e:
            # print(f"Error reading embeddings for {row[ligand_column]} {row[ligand_column]} {row[ligand_column]}: {e}")
            idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(idx)
        
        if not self.inference:
            try:
                pocket_data = {
                    "src": self._read_pocket_coordinates(ligand_id=row[self._ligand_column], p_name=row['mov_protein']),
                    "tar": self._read_pocket_coordinates(ligand_id=row[self._ligand_column], p_name=row['ref_protein'])
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
        atomic_plus_residue_embedding = np.concatenate([atom_embeddings, esm_per_atom], axis=-1)
        # atomic_plus_residue_embedding = esm_per_atom
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

    def _read_pocket_coordinates(self, ligand_id: str, p_name: str) -> torch.Tensor:
        """
        Reads pocket CA coordinates and their residue indices from a single saved pickle file.

        Args:
            ligand_id (str): Identifier for the ligand.
            p_name (str): Protein name.

        Returns:
            torch.Tensor: Tensor of residue indices for the pocket.
        Raises:
            ValueError: If the pickle file format is invalid or missing expected keys.
            Exception: If there is an error loading the pickle file.
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

            residue_indices = torch.tensor(data['residue_indices'], dtype=torch.int64)

            return residue_indices

        except Exception as e:
            logging.error(f"Error loading pocket data from {pickle_path}: {e}")
            raise

    def extract_esm_embeddings(self, pdb_file: str, chain_name: str) -> dict:
        """
        Extract ESM embeddings for all standard residues in a PDB file, with caching.
        
        Returns:
            Dict mapping resseq (int) → ESM embedding tensor.
        """

        # === Caching ===


        pdb_hash = hashlib.md5(pdb_file.encode()).hexdigest()
        cache_path = os.path.join(self._cache_dir, f"{pdb_hash}.pt")
        # if os.path.exists(cache_path):
        #     return torch.load(cache_path)
        
        # === Compute ESM embeddings ===
        structure = self._mmcif_parser.get_structure("pdb", pdb_file)
        
        model = list(structure)[0]
        chain = list(model)[0]
        peptides = self._ppb_builder.build_peptides(chain)

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
        # torch.save(embedding_dict, cache_path)
        
        fasta_cache_dir = os.path.join(self._base_data_path, "fasta")
        os.makedirs(fasta_cache_dir, exist_ok=True)
        fasta_path = os.path.join(fasta_cache_dir, f"{chain_name}.fasta")
        if not os.path.exists(fasta_path):
            with open(fasta_path, "w") as f:
                f.write(f">{pdb_hash}\n")
                # Wrap sequence every 60 chars for readability
                for i in range(0, len(sequence), 60):
                    f.write(sequence[i:i+60] + "\n")

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