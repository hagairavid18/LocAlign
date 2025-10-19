import torch
from torch import nn
import torch.nn.functional as F

from models.utils.math import compute_rmsd_torch


def rotation_loss_frobenius(R1, R2):
    # R1 and R2 are the rotation matrices with shape (B, 3, 3)
    loss = torch.norm(R1 - R2, p='fro', dim=(1, 2))
    return loss.mean()

class RTLoss(nn.Module):
    def __init__(self, translation_weight: float = 0.01):
        super(RTLoss, self).__init__()
        self._translation_weight = translation_weight
    
    def forward(self, batch, rotation_ab_pred, translation_ab_pred):
        rotation_mse = rotation_loss_frobenius(rotation_ab_pred, batch['gt_R'])
        translation_mse = F.mse_loss(translation_ab_pred, batch['gt_t'][:, :3])
        translation_rmse = torch.sqrt(translation_mse)
        return {'transformation': rotation_mse + self._translation_weight * translation_rmse, 'rotation': rotation_mse, 'translation': translation_rmse}


class PocketLoss(nn.Module):
    def __init__(self, return_non_linear: bool = False, alpha: float = 1.0):
        super(PocketLoss, self).__init__()
        self._return_non_linear = return_non_linear
        self._alpha = alpha
    
    def forward(self, batch, rotation_ab_pred, translation_ab_pred):
        pocket_coordinates = batch['src_pocket_frames'][:, :, 0, :]
        pocket_rmsd = compute_rmsd_torch(pocket_coordinates, batch['gt_R'], batch['gt_t'], rotation_ab_pred, translation_ab_pred, batch['src_pocket_mask']).mean()
        if self._return_non_linear:
            non_linear_pocket_rmsd = pocket_rmsd / (pocket_rmsd + self._alpha ** 2)
        print(f"Pocket RMSD: {pocket_rmsd.item()}")
        return {'non_linear_pocket_rmsd': non_linear_pocket_rmsd, "pocket_rmsd": pocket_rmsd}


class LigandLoss(nn.Module):
    def __init__(self, return_non_linear: bool = False, alpha: float = 1.0):
        super(LigandLoss, self).__init__()
        self._return_non_linear = return_non_linear
        self._alpha = alpha
    
    def forward(self, batch, rotation_ab_pred, translation_ab_pred):
        src_ligand_coordiantes = batch['src_ligand_coordinates']
        tar_ligand_coordiantes = batch['tar_ligand_coordinates']
        mask = batch['src_ligand_mask']
        src_ligand_coordiantes_transformed = torch.matmul(src_ligand_coordiantes, rotation_ab_pred) + translation_ab_pred[:,:3].unsqueeze(1)
        
        squared_diff = torch.sum((tar_ligand_coordiantes - src_ligand_coordiantes_transformed) ** 2, dim=2)  
        masked_squared_diff = squared_diff * mask.float()  
        valid_counts = mask.sum(dim=1)  

        rmsd_value = torch.sqrt(masked_squared_diff.sum(dim=1) / valid_counts.clamp(min=1e-10))  
        return {"ligand_rmsd": rmsd_value.mean()}


class CentroidLigandLoss(nn.Module):
    def __init__(self):
        super(CentroidLigandLoss, self).__init__()
    
    def forward(self, batch, rotation_ab_pred, translation_ab_pred):
        src_ligand_coordiantes = batch['src_ligand_coordinates']
        tar_ligand_coordiantes = batch['tar_ligand_coordinates']
        mask = batch['src_ligand_mask']
        src_transformed = torch.matmul(src_ligand_coordiantes, rotation_ab_pred) + translation_ab_pred[:,:3].unsqueeze(1)
        
        src_transformed_centroid = (src_transformed * mask.unsqueeze(2).float()).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1e-10)
        tar_centroid = (tar_ligand_coordiantes * mask.unsqueeze(2).float()).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1e-10)
        
        
        squared_diff = torch.sum((tar_centroid - src_transformed_centroid) ** 2, dim=1)
        rmsd_value = torch.sqrt(squared_diff)
        
        return {"centroid_ligand_rmsd": rmsd_value.mean()}