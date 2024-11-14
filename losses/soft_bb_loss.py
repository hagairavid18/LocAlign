import torch
from torch import nn
import torch.nn.functional as F

from objects.protein_pair import ProteinPair


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
        return {'transformation': rotation_mse + self._translation_weight * translation_mse, 'rotation': rotation_mse, 'translation': translation_mse}

class PocketLoss(nn.Module):
    def __init__(self, return_non_linear: bool = False, alpha: float = 1.0):
        super(PocketLoss, self).__init__()
        self._return_non_linear = return_non_linear
        self._alpha = alpha
    
    def forward(self, batch, rotation_ab_pred, translation_ab_pred):
        pocket_atoms = batch['src_pocket']
        pocket_rmsd = ProteinPair.compute_rmsd_torch(pocket_atoms, batch['gt_R'], batch['gt_t'], rotation_ab_pred, translation_ab_pred, batch['src_pocket_mask'])
        if self._return_non_linear:
            pocket_rmsd = pocket_rmsd / (pocket_rmsd + self._alpha ** 2)
        return {'pocket_rmsd': pocket_rmsd.mean()}