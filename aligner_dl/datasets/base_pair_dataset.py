import json
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.misc import deserialize_nested_lists
from utils.constants import ALL_INVALID_LIGANDS

import random
import numpy as np

# Set random seed for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BasePairDataset(Dataset):
    def __init__(
            self, 
            df_path: str, 
            base_data_path: str, 
            base_embedding_path: str, 
            n_samples: int,
            ligand_column: str = 'Ligand_ID', 
            min_cath: int = 0, 
            max_cath: int = 8, 
            only_one_transformation: bool = True, 
            bbc_filter_ratio = 0.0, seed: int | None = None, 
            inference: bool= False
            ) -> None:
        self._df_path = df_path
        self._base_data_path = base_data_path
        self._base_embedding_path = base_embedding_path
        self._n_samples = n_samples
        self._only_one_transformation = only_one_transformation
        self._min_cath = min_cath
        self._max_cath = max_cath
        self._ligand_column = ligand_column
        self.inference = inference 
        assert max_cath >= min_cath, f"max_cath ({max_cath}) must be greater than min_cath ({min_cath})"
        self._bbc_filter_ratio = bbc_filter_ratio
        if seed:
            set_seed(seed)

        self._df: pd.DataFrame = self._read_data_path()
                
    def _calculate_sample_weights(self, df):
        ligand_counts = df['Ligand_ID'].value_counts()
        weights = 1 / (ligand_counts ** 0.5)

        weights = weights / weights.sum()
        weights = np.clip(weights, 0.1, 1)

        df['sample_weight'] = df['Ligand_ID'].apply(lambda x: weights[x])
        df['sample_weight'] = 1
        return df

    def _read_data_path(self) -> pd.DataFrame:
        assert os.path.exists(self._df_path), f"Can't find path {self._df_path}"
        pairs = pd.read_csv(self._df_path)
        if 'index' in pairs.columns:
            pairs = pairs.drop('index', axis=1)
        if not self.inference:
            if self._only_one_transformation:
                pairs = pairs[pairs['n_transformations'] == 1]
            pairs = pairs[pairs['cath_degree'] >= self._min_cath].reset_index()
            pairs = pairs[pairs['cath_degree'] <= self._max_cath].reset_index()

            if self._n_samples:
                pairs = pairs.sample(n=self._n_samples, random_state=42, replace=True)
            pairs = pairs[(pairs['cath_degree'] >= 4) | ((pairs['cath_degree'] < 4) & (pairs['bbc'] > self._bbc_filter_ratio))]
            print(f"Read df with {len(pairs)} pairs")
        
        for col in pairs.columns:
            if pairs[col].apply(lambda x: isinstance(x, str) and x.startswith('[') and x.endswith(']')).any():
                pairs[col] = pairs[col].fillna('[]')
                
                pairs[col] = pairs[col].apply(lambda x: json.loads(x))
                pairs[col] = pairs[col].apply(lambda x: deserialize_nested_lists(x, col))
             

        # pairs = self._calculate_sample_weights(pairs)
        # test whether some ligands are invalid
        invalid_ligands = set(pairs[self._ligand_column].unique()).intersection(ALL_INVALID_LIGANDS)
        if invalid_ligands:
            print(f"Warning: Found invalid ligands in the dataset: {invalid_ligands}")
            pairs = pairs[~pairs[self._ligand_column].isin(invalid_ligands)].reset_index(drop=True)
            print(f"After removing invalid ligands, {len(pairs)} pairs remain.")

        return pairs

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label
