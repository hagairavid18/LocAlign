import json
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.loading import deserialize_nested_lists

class BasePairDataset(Dataset):
    def __init__(self, df_path: str, base_data_path: str, n_samples: int):
        self._df_path = df_path
        self._base_data_path = base_data_path
        self._n_samples = n_samples

        self._df: pd.DataFrame = self._read_data_path()

    def _read_data_path(self) -> pd.DataFrame:
        assert os.path.exists(self._df_path), f"Can't find path {self._df_path}"
        pairs = pd.read_csv(self._df_path)[:self._n_samples]
        for col in pairs.columns:
            if pairs[col].apply(lambda x: isinstance(x, str) and x.startswith('[') and x.endswith(']')).any():
                pairs[col] = pairs[col].fillna('[]')
                
                pairs[col] = pairs[col].apply(lambda x: json.loads(x))
                pairs[col] = pairs[col].apply(lambda x: deserialize_nested_lists(x, col))
        print(f"Read df with {len(pairs)} pairs")
        pairs['HardBBS_rotations'] = None
        pairs['HardBBS_translations'] = None

        return pairs

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label
