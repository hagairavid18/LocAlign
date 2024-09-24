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
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_batch_to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(move_batch_to_device(v, device) for v in batch)
    else:
        return batch  # For non-tensor types, return as is