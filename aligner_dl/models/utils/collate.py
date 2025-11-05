import torch


def custom_collate_fn(batch: list[dict[torch.Tensor, dict]]):
    # Assume batch is a list of dictionaries
    batch_dict = {}
    for key, val in batch[0].items():
        # Stack all tensors for a given key
        if isinstance(val, dict):
            batch_dict[key] = [item[key] for item in batch]
        elif isinstance(val, int):
            batch_dict[key] = [item[key] for item in batch]
        else:
            batch_dict[key] = torch.stack([item[key] for item in batch])
            
    return batch_dict


import torch
from collections.abc import Mapping, Sequence

def move_batch_to_device(batch, device, *, non_blocking=True, cast_inputs_for_amp=False, amp_dtype=torch.bfloat16, exclude_keys=frozenset({"labels", "target", "targets"})):
    """Move a nested batch to device. Optionally cast *floating* inputs for bandwidth savings."""
    amp_on = torch.is_autocast_enabled()

    def should_cast(key_path, t: torch.Tensor):
        if not cast_inputs_for_amp or not amp_on:
            return False
        if not t.is_floating_point() or t.dtype != torch.float32:
            return False
        # don't cast common target/label fields
        return not any(k in exclude_keys for k in key_path)

    def _move(x, key_path=()):
        # Tensors
        if isinstance(x, torch.Tensor):
            # move first (non_blocking if from pinned CPU)
            y = x if x.device == device else x.to(device, non_blocking=non_blocking)
            # optional pre-cast for AMP (inputs only)
            if should_cast(key_path, y):
                if y.dtype != amp_dtype:
                    y = y.to(dtype=amp_dtype)
            return y

        # PyG Data or any object with a .to(device) method
        if hasattr(x, "to") and callable(getattr(x, "to")) and not isinstance(x, (str, bytes)):
            try:
                return x.to(device, non_blocking=non_blocking)
            except TypeError:
                return x.to(device)

        # Mappings (dict-like)
        if isinstance(x, Mapping):
            return type(x)({k: _move(v, key_path + (k,)) for k, v in x.items()})

        # Sequences (list/tuple) but not strings/bytes
        if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
            return type(x)(_move(v, key_path) for v in x)

        # Leave everything else as is
        return x

    return _move(batch)

