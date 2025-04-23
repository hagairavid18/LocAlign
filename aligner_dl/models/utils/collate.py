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


def move_batch_to_device(batch, device):
    amp_enabled = torch.is_autocast_enabled()
    target_dtype = torch.bfloat16 if amp_enabled else None

    def move_and_cast(x):
        if isinstance(x, torch.Tensor):
            if target_dtype is not None and x.dtype == torch.float32:
                return x.to(device=device, dtype=target_dtype)
            else:
                return x.to(device)
        return x

    if isinstance(batch, dict):
        return {k: move_and_cast(v) if not isinstance(v, (dict, list, tuple)) else move_batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_batch_to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(move_batch_to_device(v, device) for v in batch)
    else:
        return move_and_cast(batch)
