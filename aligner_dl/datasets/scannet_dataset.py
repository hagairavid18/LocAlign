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


class ScanNetDataset(BasePairDataset):
    _MAX_LIGAND_LENGTH = 500

    def __init__(
            self, 
            df_path: str, 
            base_scannet_path: str = None, 
            base_data_path: str = LIGAND_DIR, 
            n_samples: int | None = None, 
            min_cath: int = 0, 
            max_cath: int = 8, 
            bbc_filter_ratio: float = 0.0, 
            max_length: int = 2000, 
            seed: int| None = None, 
            inference: bool = False, 
            ligand_column: str = 'Ligand_ID',
            esm_model: str = None,
            base_esm_embedding_path: str = LIGAND_DIR,
            use_esm: bool = True,
            esm_layer: int = 28
            ) -> None:
        """
        Initializes the ScanNetDataset.

        Args:
            df_path (str): _path to the CSV file containing the dataset metadata.
            base_scannet_path (str): 
            base_data_path (str, optional): _description_. Defaults to LIGAND_DIR.
            n_samples (int | None, optional): _description_. Defaults to None.
            min_cath (int, optional): _description_. Defaults to 0.
            max_cath (int, optional): _description_. Defaults to 8.
            bbc_filter_ratio (float, optional): _description_. Defaults to 0.0.
            seed (int | None, optional): _seed for random operations. Defaults to None.
            max_length (int, optional): _maximum number of atoms to consider. Defaults to 2000.
            inference (bool, optional): Whether the dataset is used for inference. Defaults to False.
            ligand_column (str, optional): Column name for ligand IDs in the dataframe. Defaults to 'Ligand_ID'.
            esm_model (str, optional): Name of the ESM model to use. Defaults to None.
            use_esm (bool, optional): Whether to use ESM embeddings. Defaults to True.
            esm_layer (int, optional): Layer of the ESM model to extract embeddings from.
        """        
        super().__init__(
            df_path, 
            base_data_path, 
            n_samples,
            ligand_column=ligand_column, 
            min_cath=min_cath, 
            max_cath=max_cath, 
            seed=seed, 
            bbc_filter_ratio=bbc_filter_ratio, 
            inference=inference)
        
        self._mmcif_parser = PDBParser()
        self._max_atoms = max_length 
        self._scannet_dir = base_scannet_path
                
        self._with_esm = use_esm
        if esm_model is not None:
            self._esm_layer= esm_layer
            self._init_esm_model(esm_model, base_esm_embedding_path)

    def _init_esm_model(self, esm_model: str, base_esm_embedding_path: str = None) -> None:
        self._esm_model, self._esm_alphabet = getattr(esm.pretrained, esm_model)()
        self._batch_converter = self._esm_alphabet.get_batch_converter()
        self._esm_model = self._esm_model.eval()
        self._ppb_builder = PPBuilder()
        base_esm_path = base_esm_embedding_path if base_esm_embedding_path is not None else self._base_data_path
        self._cache_dir = os.path.join(base_esm_path, f"esm_cache_{esm_model}_{self._esm_layer}")
        os.makedirs(self._cache_dir, exist_ok=True)

    def __getitem__(self, idx: int) -> dict[torch.Tensor]:
        row = self._df.iloc[idx]

        try:
            esm_embeddings_tar = self.extract_esm_embeddings(
                pdb_file=os.path.join(self._base_data_path, row[self._ligand_column], row['ref_protein'] + row['ref_chain'] + '_non_ligand_.ent'),
                chain_name=row['ref_protein'] + '_' + row['ref_chain']
            )
            esm_embeddings_src = self.extract_esm_embeddings(
                pdb_file=os.path.join(self._base_data_path, row[self._ligand_column], row['mov_protein'] + row['mov_chain'] + '_non_ligand_.ent'),
                chain_name=row['mov_protein'] + '_' + row['mov_chain']
            ) 
            embedding_dicts = {
                "tar": self._read_embedding(ligand_id=row[self._ligand_column], chain=row['ref_protein'] + row['ref_chain'], esm_embedding_dict=esm_embeddings_tar),
                "src": self._read_embedding(ligand_id=row[self._ligand_column], chain=row['mov_protein'] + row['mov_chain'], esm_embedding_dict=esm_embeddings_src),
            }
            # print(f"Read embeddings for {row[self._ligand_column]} {row['mov_protein']} {row['ref_protein']}")
            if not self.inference:
                src_ligand_coordinates, src_atom_ids = self._read_ligand(ligand_id=row[self._ligand_column], chain=row['mov_protein'] + row['mov_chain'])
                tar_ligand_coordinates, tar_atom_ids = self._read_ligand(ligand_id=row[self._ligand_column], chain=row['ref_protein'] + row['ref_chain'])
                if not src_atom_ids == tar_atom_ids:
                    shared_atom_ids = set(src_atom_ids).intersection(tar_atom_ids)
                    src_ligand_coordinates = src_ligand_coordinates[torch.tensor([src_atom_ids.index(atom_id) for atom_id in shared_atom_ids])]
                    tar_ligand_coordinates = tar_ligand_coordinates[torch.tensor([tar_atom_ids.index(atom_id) for atom_id in shared_atom_ids])]
                    if src_ligand_coordinates.shape != tar_ligand_coordinates.shape or src_ligand_coordinates.shape[0] == 0:
                        raise ValueError("Mismatched ligand coordinates after filtering to shared atoms")

        except Exception as e:
            print(f"Error reading embeddings for {row['ref_protein']} {row['mov_protein']}: {e}")
            idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(idx)

        ret = {}
        for key in ["src", "tar"]:
            embedding_dict = embedding_dicts[key]
            n_atoms = embedding_dict['atom_embeddings'].shape[0]

            ret[f'{key}_pretrained_embeddings'] = F.pad(embedding_dict[f'atom_embeddings'], (0, 0, 0, self._max_atoms - n_atoms))
            ret[f'{key}_frames'] = F.pad(embedding_dict[f'atom_frames'], (0, 0, 0, 0, 0, self._max_atoms - n_atoms))
            ret[f'{key}_neighbors'] = F.pad(embedding_dict[f'atom_neighbors'], (0, 0, 0, self._max_atoms - n_atoms), value=-1)
            ret[f'{key}_residue_indices'] = F.pad(embedding_dict[f'atom_residue_indices'], (0, self._max_atoms - n_atoms))
            ret[f'{key}_atom_original_indices'] = F.pad(embedding_dict[f'atom_original_indices'], (0, self._max_atoms - n_atoms))
            ret[f'{key}_mask'] = F.pad(torch.ones(n_atoms), (0, self._max_atoms - n_atoms), value=0).bool()

        ret['metadata'] = row.to_dict()
        if self.inference:
            return ret
        

        if len(src_ligand_coordinates) > self._MAX_LIGAND_LENGTH or len(src_ligand_coordinates) == 0:
            print(f"Source ligand length {len(src_ligand_coordinates)} exceeds max length {self._MAX_LIGAND_LENGTH} or is zero")
            idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(idx)
        
        ret['src_ligand_coordinates'] = F.pad(src_ligand_coordinates, (0, 0, 0, self._MAX_LIGAND_LENGTH - len(src_ligand_coordinates)))
        ret['tar_ligand_coordinates'] = F.pad(tar_ligand_coordinates, (0, 0, 0, self._MAX_LIGAND_LENGTH - len(tar_ligand_coordinates)))

        ret['src_ligand_mask'] = F.pad(torch.ones(len(src_ligand_coordinates)), (0, self._MAX_LIGAND_LENGTH - len(src_ligand_coordinates)), value=0).bool()
        ret['tar_ligand_mask'] = F.pad(torch.ones(len(tar_ligand_coordinates)), (0, self._MAX_LIGAND_LENGTH - len(tar_ligand_coordinates)), value=0).bool()
        
        return ret

    def _read_embedding(
        self, 
        chain: str, 
        ligand_id: str,
        esm_embedding_dict: dict[int, torch.Tensor] | None = None
        ) -> tuple[torch.Tensor, torch.Tensor]:
        scannet_embedding_path = os.path.join(self._scannet_dir, ligand_id,  chain + '_scannet_atoms.pkl')
        if not os.path.exists(scannet_embedding_path):
            logger.info(f"Can't find embedding path for ligand: {ligand_id} protein: {chain}")
            raise ValueError(
                f"Can't find scannet embedding path for ligand: {ligand_id} protein: {chain}"
            )
        with open(scannet_embedding_path, 'rb') as f:
            data = pickle.load(f)
        
        residue_embeddings = data["residue_embeddings"]
        residue_ids = data["residue_ids"]
        atom_embeddings = data["atomic_plus_residue_embedding"]
        atom_residue_index = data["sequence_indices_atom"]  # Residue index for each atom
        atom_frames = data["atomic_frames"]
        atom_neighbors = data["atom_nearest_neighbors"]

        # oringal_atom_indices = np.arange(len(atom_embeddings))

        residue_indices = residue_ids[:, -1].astype(int)  # Extract residue indices (last column of residue_ids)
        atom_residue_index = residue_indices[atom_residue_index]
        valid_residue_indices = set(atom_residue_index).intersection(set(esm_embedding_dict.keys()))
        
        valid_mask = np.isin(atom_residue_index, list(valid_residue_indices))
        kept_idx = np.nonzero(valid_mask)[0]  # original indices to keep

        # --- Subsample to at most max_atoms ---
        max_atoms = self._max_atoms 
        if len(kept_idx) > max_atoms:
            kept_idx = np.random.choice(kept_idx, size=max_atoms, replace=False)

        # Now build the original->new index map for only those kept
        N = len(atom_residue_index)
        orig2new = np.full(N, -1, dtype=np.int32)
        orig2new[kept_idx] = np.arange(len(kept_idx), dtype=np.int32)

        # Filter per-atom arrays
        atom_embeddings = atom_embeddings[kept_idx]
        atom_frames = atom_frames[kept_idx]
        atom_residue_index = atom_residue_index[kept_idx]

        # Remap and filter neighbors
        remapped_neighbors = orig2new[atom_neighbors]
        remapped_neighbors = remapped_neighbors[kept_idx]

        # Compact neighbors (valid first, -1s at the end)
        valid = remapped_neighbors >= 0
        order = np.argsort(~valid, axis=1)
        atom_neighbors = np.take_along_axis(remapped_neighbors, order, axis=1)


        if self._with_esm:
            esm_per_atom = np.stack([esm_embedding_dict[int(id)] for id in atom_residue_index])

            # === Concatenate ===
            atomic_plus_residue_embedding = np.concatenate([atom_embeddings, esm_per_atom], axis=-1)
        else:
            atomic_plus_residue_embedding = atom_embeddings
        

        ret_dict = {
            'atom_frames': atom_frames,
            'atom_embeddings': atomic_plus_residue_embedding,
            'residue_embeddings': residue_embeddings,
            'residue_residue_indices': residue_indices,
            'atom_residue_indices': atom_residue_index,
            'atom_neighbors': atom_neighbors,
            "atom_original_indices": kept_idx
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

    def extract_esm_embeddings(self, pdb_file: str, chain_name: str) -> dict:
        """
        Extract ESM embeddings for all standard residues in a PDB file, with caching.
        
        Returns:
            Dict mapping resseq (int) → ESM embedding tensor.
        """

        # === Caching ===
        pdb_hash = hashlib.md5(pdb_file.split('/')[-1].encode()).hexdigest()
        cache_path = os.path.join(self._cache_dir, f"{pdb_hash}.pt")
        if os.path.exists(cache_path):
            return torch.load(cache_path)
        
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
            results = self._esm_model(tokens, repr_layers=[self._esm_layer], return_contacts=False)

        reps = results["representations"][self._esm_layer][0, 1:len(sequence)+1]

        embedding_dict = {
            resseq: reps[idx].cpu()  # Save as CPU tensors for portability
            for resseq, idx in resseq_to_index.items()
        }

        # === Save to cache ===
        torch.save(embedding_dict, cache_path)
        
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

    data_path = 'LocAlign/baseline_results/2024-07-11_10-55-38_57.csv'
    data_path = 'LocAlign/baseline_results/2024-07-17_16-01-08_3000.csv'
    base_data_path = LIGAND_DIR


    # Create dataset and DataLoader
    dataset = ScanNetDataset(data_path, base_data_path)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Iterate through the DataLoader
    print("Testing DataLoader:")
    for batch_idx, (samples, labels, infos) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print("Samples:", samples)
        print("Labels:", labels)
        print("Infos:", infos)
        print()