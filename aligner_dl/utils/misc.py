import shutil

import numpy as np


def deserialize_nested_lists(data, col_name):
    if col_name == "rotations":
        return [[np.array(trans) for trans in res_trans] for res_trans in data] 
    elif col_name == 'translations':
        return [np.array(trans) for trans in data] 
    elif col_name in ['DaliAligner_rotations', 'TMAligner_rotations']:
        return np.array(data)
    elif col_name in ['DaliAligner_translations', 'TMAligner_translations']:
        return np.array(data)
    else:
        return data
    

def serialize_nested_lists(data):

    if isinstance(data, list):
        return [serialize_nested_lists(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()  # Convert ndarray to list for JSON serialization
    else:
        return data


def find_binary(name: str) -> str:
    """Absolute path of the executable `name` found on PATH; raises FileNotFoundError if missing."""
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f'{name} was not found on PATH; install it or add it to PATH')
    return path
