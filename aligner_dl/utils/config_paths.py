import os

from utils.constants import (
    DATA_ROOT,
    DATA_ROOT_ENV,
    EXTERNAL_PATH_PREFIX,
    EXTERNAL_ROOT,
    EXTERNAL_ROOT_ENV,
    MUST_EXIST_CONFIG_KEYS,
    PATH_CONFIG_KEYS,
)


def resolve_path(path: str) -> str:
    """Resolve a config path to an absolute one.

    Absolute paths are returned unchanged. A path starting with EXTERNAL_PATH_PREFIX is
    resolved against EXTERNAL_ROOT, any other relative path against DATA_ROOT.
    """
    if os.path.isabs(path):
        return path
    if path.startswith(EXTERNAL_PATH_PREFIX):
        return str(EXTERNAL_ROOT / path[len(EXTERNAL_PATH_PREFIX):])
    return str(DATA_ROOT / path)


def resolve_config_paths(config):
    """Resolve, in place, every value of the keys in PATH_CONFIG_KEYS in a loaded YAML config.

    Walks nested dicts and lists. For the keys in MUST_EXIST_CONFIG_KEYS, raises
    FileNotFoundError when the resolved path does not exist, naming the environment
    variables that relocate the data. Returns `config`.
    """
    if isinstance(config, dict):
        for key, value in config.items():
            if key in PATH_CONFIG_KEYS and isinstance(value, str):
                config[key] = resolve_path(value)
                if key in MUST_EXIST_CONFIG_KEYS and not os.path.exists(config[key]):
                    raise FileNotFoundError(
                        f'{key}: {config[key]} does not exist (data root {DATA_ROOT}, external root {EXTERNAL_ROOT}; '
                        f'set {DATA_ROOT_ENV} or {EXTERNAL_ROOT_ENV} to relocate)'
                    )
            else:
                resolve_config_paths(value)
    elif isinstance(config, list):
        for item in config:
            resolve_config_paths(item)
    return config
