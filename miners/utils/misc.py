import importlib
import json
import os
from typing import Any
import numpy as np
import pandas as pd

from miners.utils.loading import serialize_nested_lists


def build_object(config: dict, default_module: str|None = None) -> Any:
    """build a general object dynamically from config dict

    Args:
        config (Dict): a config dict with the following keys: 'name', 'args' and optionally 'module'
        default_module (Optional[str], optional): The default module to import if no module
         is specified in config. Defaults to None.

    Returns:
        Any: the object initialized
    """
    module_name = config.get('module', default_module)
    if module_name is  None:
         print("Module name is None! must specify module name in config dict or pass a default module!")
         return None
    return getattr(importlib.import_module(module_name), config['name'])(**config.get('args', {}))

def save_results_to_csv(results: list[tuple] | pd.DataFrame, start_time: str, base_dir: str = "temp_baseline", prev_results: pd.DataFrame |  None = None, save_path: str = None) -> None:
    df = results if isinstance(results, pd.DataFrame) else  pd.DataFrame([obj.__dict__ for obj in results])
    
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
            df[col] = df[col].apply(lambda x: json.dumps(serialize_nested_lists(x)))
            
    if prev_results is not None:
        df = pd.concat([prev_results, df], ignore_index=True)
    if save_path is None:
        save_dir = os.path.join("results", base_dir)
        os.makedirs(save_dir , exist_ok=True)
        save_path = f'{save_dir}/{start_time}_{df.shape[0]}.csv'
    df.to_csv(save_path, index=False)
    return df


def flatten_dict(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)