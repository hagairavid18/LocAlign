import numpy as np
import torch


def euclidean_to_spherical(x: torch.Tensor, cut='2pi', eps=1e-8) -> torch.Tensor:
    """
    Convert Euclidean coordinates to spherical coordinates.
    Args:
        x (torch.Tensor): Input tensor of shape (..., 3) representing Euclidean coordinates.
        cut (str): If '2pi', the azimuthal angle phi is adjusted to be in the range [0, 2π].
        eps (float): Small value to avoid division by zero.
    Returns:
        torch.Tensor: Tensor of shape (..., 3) representing spherical coordinates (r, theta, phi).
    """
    if x.shape[-1] != 3:
        raise ValueError("Input tensor must have the last dimension of size 3 for spherical coordinates conversion.")
    r = torch.linalg.norm(x, dim=-1)
    theta = torch.acos(x[..., 2] / (r + eps))
    phi = torch.atan2(x[..., 1], x[..., 0] + eps)
    
    # if cut == '2pi':
        # phi = phi + (phi < 0).float() * (2 * np.pi)
    
    return torch.stack([r, theta, phi], dim=-1)


def compute_rmsd_torch(coordinates: torch.Tensor, gt_R: torch.Tensor, gt_t: torch.Tensor, pred_R: torch.Tensor, pred_t:torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

    transformed_points_1 = torch.matmul(coordinates, gt_R) + gt_t[:,:3].unsqueeze(1)
    
    transformed_points_2 = torch.matmul(coordinates, pred_R) + pred_t.unsqueeze(1)
    
    squared_diff = torch.sum((transformed_points_1 - transformed_points_2) ** 2, dim=2)  # Shape (B, N)
    masked_squared_diff = squared_diff * mask.float()  # Shape [B, N], mask applied

    valid_counts = mask.sum(dim=1)  # Shape [B]

    rmsd_value = torch.sqrt(masked_squared_diff.sum(dim=1) / valid_counts.clamp(min=1e-10))  # Shape [B]
            
    return rmsd_value


def compose_transformations(rotations: list[torch.Tensor], translations: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compose the transformations to get the final rotation and translation matrices.

        Args:
            rotations (list[torch.Tensor]): Residual rotation from each step.
            translations (list[torch.Tensor]): residual translation from each step.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The final rotation and translation matrices.
        """
        batch_size = rotations[0].shape[0]
        device = rotations[0].device        
        R_total = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1) 
        t_total = torch.zeros(batch_size, 3, device=device)

        for R, t in zip(rotations, translations):
            R_total = R_total @ R
            t_total = torch.bmm(t_total.unsqueeze(1), R).squeeze(1) + t

        return R_total, t_total


def group_lasso_regularization(tensor: torch.Tensor, lambda_gl: float = 1e-3) -> torch.Tensor:
        """
        Computes Group Lasso regularization for a B x N x 3 tensor.
        - Sums over B first, then N, then computes sqrt of the sum of squares across 3.
        """
        reg_loss = torch.sqrt(torch.sum(tensor ** 2, dim=(0, 1)) + 1e-4)  # Sum over B, then N
        return lambda_gl * reg_loss.mean()