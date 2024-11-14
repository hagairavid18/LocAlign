import json
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.loading import deserialize_nested_lists

import random
import numpy as np

# Set random seed for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BasePairDataset(Dataset):
    def __init__(self, df_path: str, base_data_path: str, n_samples: int, min_cath: int = 0, only_one_transformation:bool = True, seed: int | None = None):
        self._df_path = df_path
        self._base_data_path = base_data_path
        self._n_samples = n_samples
        self._only_one_transformation = only_one_transformation
        self._min_cath = min_cath
        if seed:
            set_seed(seed)

        self._df: pd.DataFrame = self._read_data_path()

    def _read_data_path(self) -> pd.DataFrame:
        assert os.path.exists(self._df_path), f"Can't find path {self._df_path}"
        pairs = pd.read_csv(self._df_path)
        if 'index' in pairs.columns:
            pairs = pairs.drop('index', axis=1)
        if self._only_one_transformation:
            pairs = pairs[pairs['n_transformations'] == 1]
        pairs = pairs[pairs['cath_degree'] >= self._min_cath].reset_index()

        if self._n_samples:
            pairs = pairs.reset_index().sample(n=self._n_samples, random_state=42, replace=True)
        for col in pairs.columns:
            if pairs[col].apply(lambda x: isinstance(x, str) and x.startswith('[') and x.endswith(']')).any():
                pairs[col] = pairs[col].fillna('[]')
                
                pairs[col] = pairs[col].apply(lambda x: json.loads(x))
                pairs[col] = pairs[col].apply(lambda x: deserialize_nested_lists(x, col))
        print(f"Read df with {len(pairs)} pairs")

        return pairs

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label
