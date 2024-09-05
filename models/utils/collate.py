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