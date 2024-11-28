import torch


def compute_rmsd_torch(coordinates: torch.Tensor, gt_R: torch.Tensor, gt_t: torch.Tensor, pred_R: torch.Tensor, pred_t:torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

    transformed_points_1 = torch.matmul(coordinates, gt_R) + gt_t[:,:3].unsqueeze(1)
    
    transformed_points_2 = torch.matmul(coordinates, pred_R) + pred_t.unsqueeze(1)
    
    squared_diff = torch.sum((transformed_points_1 - transformed_points_2) ** 2, dim=2)  # Shape (B, N)
    masked_squared_diff = squared_diff * mask.float()  # Shape [B, N], mask applied

    valid_counts = mask.sum(dim=1)  # Shape [B]

    rmsd_value = torch.sqrt(masked_squared_diff.sum(dim=1) / valid_counts.clamp(min=1e-10))  # Shape [B]
            
    return rmsd_value