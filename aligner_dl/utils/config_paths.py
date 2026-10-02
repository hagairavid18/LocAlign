import os

from utils.constants import (
    DATA_ROOT,
    DATA_ROOT_ENV,
    MUST_EXIST_CONFIG_KEYS,
    PATH_CONFIG_KEYS,
    PATH_PLACEHOLDERS,
)


def resolve_path(path: str) -> str:
    """Resolve a config path to an absolute one.

    Every ${NAME} with NAME in PATH_PLACEHOLDERS (for example ${SCANNET_DIR}) is replaced by
    the corresponding constant. Absolute paths are then returned unchanged and relative paths
    are resolved against DATA_ROOT.
    """
    for name, value in PATH_PLACEHOLDERS.items():
        path = path.replace('${' + name + '}', value)
    if os.path.isabs(path):
        return path
    return str(DATA_ROOT / path)


def resolve_config_paths(config):
    """Resolve, in place, every value of the keys in PATH_CONFIG_KEYS in a loaded YAML config.

    Walks nested dicts and lists. For the keys in MUST_EXIST_CONFIG_KEYS, raises
    FileNotFoundError when the resolved path does not exist, naming the environment
    variable that relocates the data. Returns `config`.
    """
    if isinstance(config, dict):
        for key, value in config.items():
            if key in PATH_CONFIG_KEYS and isinstance(value, str):
                config[key] = resolve_path(value)
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
