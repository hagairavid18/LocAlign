import os
from pathlib import Path

from utils.constants import (
    CHECKOUT_ROOT,
    DATA_ROOT,
    DATA_ROOT_ENV,
    MUST_EXIST_CONFIG_KEYS,
    PATH_CONFIG_KEYS,
    PATH_PLACEHOLDERS,
    UNTRACKED_PATH_CONFIG_KEYS,
)


def resolve_path(
    path: str,
    root: Path,
) -> str:
    """Resolve a config path to an absolute one.

    Every ${NAME} with NAME in PATH_PLACEHOLDERS (for example ${SCANNET_DIR}) is replaced by
    the corresponding constant. Absolute paths are then returned unchanged and relative paths
    are resolved against `root`.
    """
    for name, value in PATH_PLACEHOLDERS.items():
        path = path.replace('${' + name + '}', value)
    if os.path.isabs(path):
        return path
    return str(root / path)


def resolve_config_paths(config):
    """Resolve, in place, every value of the keys in PATH_CONFIG_KEYS in a loaded YAML config.

    Relative paths resolve against CHECKOUT_ROOT (tracked files such as df_path), except the
    keys in UNTRACKED_PATH_CONFIG_KEYS (ckpt_path), which resolve against DATA_ROOT.

    Walks nested dicts and lists. For the keys in MUST_EXIST_CONFIG_KEYS, raises
    FileNotFoundError when the resolved path does not exist, naming the environment
    variable that relocates the data. Returns `config`.
    """
    if isinstance(config, dict):
        for key, value in config.items():
            if key in PATH_CONFIG_KEYS and isinstance(value, str):
                root = DATA_ROOT if key in UNTRACKED_PATH_CONFIG_KEYS else CHECKOUT_ROOT
                config[key] = resolve_path(value, root)
                if key in MUST_EXIST_CONFIG_KEYS and not os.path.exists(config[key]):
                    raise FileNotFoundError(
                        f'{key}: {config[key]} does not exist (data root {DATA_ROOT}; '
                        f'set {DATA_ROOT_ENV} to relocate)'
                    )
            else:
                resolve_config_paths(value)
    elif isinstance(config, list):
        for item in config:
            resolve_config_paths(item)
    return config
