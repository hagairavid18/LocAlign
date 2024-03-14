import importlib
from typing import Any


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
    assert module_name is not None, "Module name is None! must specify module name in config dict or pass a default module!"
    return getattr(importlib.import_module(module_name), config['name'])(**config.get('args', {}))