import torch
from torch import nn
import torch.nn.functional as F


def rotation_loss_frobenius(R1, R2):
    # R1 and R2 are the rotation matrices with shape (B, 3, 3)
    loss = torch.norm(R1 - R2, p='fro', dim=(1, 2))  # Frobenius norm over the last two dimensions
    return loss.mean()

class RTLoss(nn.Module):
    def __init__(self, translation_weight: float = 0.01):
        super(RTLoss, self).__init__()
        self._translation_weight = translation_weight
    
    def forward(self, rotation_ab_pred, translation_ab_pred, rotation_ab, translation_ab):
        rotation_mse = rotation_loss_frobenius(rotation_ab_pred, rotation_ab)
        translation_mse = F.mse_loss(translation_ab_pred, translation_ab[:, :3])
        return {'loss': rotation_mse + self._translation_weight * translation_mse, 'rot_loss': rotation_mse, 'tran_loss': translation_mse}