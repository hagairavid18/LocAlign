import torch
from torch import nn
import torch.nn.functional as F

from objects.protein_pair import ProteinPair


def rotation_loss_frobenius(R1, R2):
    # R1 and R2 are the rotation matrices with shape (B, 3, 3)
    loss = torch.norm(R1 - R2, p='fro', dim=(1, 2))  # Frobenius norm over the last two dimensions
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
    def __init__(self):
        super(PocketLoss, self).__init__()
    
    def forward(self, batch, rotation_ab_pred, translation_ab_pred):
        pocket_atoms = batch['src_pocket']
        pocket_rmsd = ProteinPair.compute_rmsd_torch(pocket_atoms, batch['gt_R'], batch['gt_t'], rotation_ab_pred, translation_ab_pred, batch['src_pocket_mask'])
        return {'pocket_rmsd': pocket_rmsd.mean()}