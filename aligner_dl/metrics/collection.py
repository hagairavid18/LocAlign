from typing import Dict, Any
from torch.nn import Module

from models.utils.misc import build_object


class MetricsCollection(Module):
    """Container for multiple metric modules.

    This class accepts a dict where the values are either already-instantiated
    metric Modules or config dicts that can be passed to `build_object` to
    construct the metric. This keeps config-driven construction simple.
    """

    def __init__(self, metrics: Dict[str, Any]):
        super().__init__()
        built = {}
        for name, cfg in metrics.items():
            if isinstance(cfg, dict) and 'name' in cfg:
                # treat as config to build
                built[name] = build_object(cfg, default_module='aligner_dl.metrics')
            else:
                built[name] = cfg
        self.metrics = built

    def reset(self):
        for m in self.metrics.values():
            if hasattr(m, 'reset'):
                m.reset()

    def update(self, batch, outputs):
        for m in self.metrics.values():
            m.update(batch, outputs)

    def compute(self) -> Dict[str, Any]:
        results = {}
        for name, m in self.metrics.items():
            res = m.compute()
            # prefix metric keys to avoid clashes
            for k, v in res.items():
                # results[f"{name}/{k}"] = v
                results[f"{k}"] = v
        return results
