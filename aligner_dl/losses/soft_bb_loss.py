import torch
from torch import nn

from models.utils.math import compute_rmsd_torch


class QualityLoss(nn.Module):
    def __init__(self, weight_dict: dict[str, float]):
        super(QualityLoss, self).__init__()
        self._weight_dict = weight_dict
        self._embedding_term = EmbeddingSimilarityLoss()
        self._weight_entropy_term = WeightEntropyLoss()

    def forward(self, outputs):
        loss_dict = {}
        corr_rmsd = outputs['corr_rmsd'].mean()
        embedding_similarity = self._embedding_term(outputs)
        gap = self._weight_entropy_term(outputs)
        loss = - self._weight_dict['embedding'] * embedding_similarity - self._weight_dict['gap'] * gap + self._weight_dict['corr_rmsd'] * corr_rmsd
        loss_dict['embedding'] = embedding_similarity
        loss_dict['gap'] = gap
        loss_dict['corr_rmsd'] = corr_rmsd
        return loss, loss_dict

class LocAlignLoss(nn.Module):
    def __init__(self, weight_dict: dict[str, float], return_non_linear: bool = True, reduce: bool = True):
        super(LocAlignLoss, self).__init__()
        self._quality_loss = QualityLoss(weight_dict)

        self._ligand_loss = LigandLoss(return_non_linear=return_non_linear)
        self._ligand_loss_weight = weight_dict.get('ligand_rmsd', 1.0)
        self._reduce = reduce

    def forward(self, batch, outputs, inference: bool = False):
        rotation_ab_pred = outputs['pred_R']
        translation_ab_pred = outputs['pred_t']
        quality_loss, quality_loss_dict = self._quality_loss(outputs)
        if inference:
            quality_loss_dict['loss'] = quality_loss
            return quality_loss, quality_loss_dict
        ligand_loss_dict = self._ligand_loss(batch, rotation_ab_pred, translation_ab_pred, reduce=self._reduce)
        total_loss = quality_loss + self._ligand_loss_weight * ligand_loss_dict['ligand_rmsd']
        loss_dict = {**quality_loss_dict, **ligand_loss_dict, 'loss': total_loss}
        return total_loss, loss_dict


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
    def __init__(self, return_non_linear: bool = False, alpha: float = 1.0, rmsd0: float = 10.0):
        super(LigandLoss, self).__init__()
        self._rmsd0 = rmsd0
        self._return_non_linear = return_non_linear
        self._alpha = alpha
    
    def forward(self, batch, rotation_ab_pred, translation_ab_pred, reduce: bool = True):
        src_ligand_coordinates = batch['src_ligand_coordinates']
        tar_ligand_coordinates = batch['tar_ligand_coordinates']
        mask = batch['src_ligand_mask']
        src_ligand_coordinates_transformed = torch.matmul(src_ligand_coordinates, rotation_ab_pred) + translation_ab_pred[:,:3].unsqueeze(1)

        squared_diff = torch.sum((tar_ligand_coordinates - src_ligand_coordinates_transformed) ** 2, dim=2)
        masked_squared_diff = squared_diff * mask.float()
        valid_counts = mask.sum(dim=1)

        rmsd_value = torch.sqrt(masked_squared_diff.sum(dim=1) / valid_counts.clamp(min=1e-10))

        if self._return_non_linear:
            rmsd_value = rmsd_value / (1 + rmsd_value/ self._rmsd0)
        if reduce:
            return {"ligand_rmsd": rmsd_value.mean()}
        else:
            return rmsd_value
        

class EmbeddingSimilarityLoss(nn.Module):
    def __init__(self):
        super(EmbeddingSimilarityLoss, self).__init__()
        # self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def _embedding_cov_term(
        self,
        top_corr_values: torch.Tensor,   # [B, K'] -> weights w_m
        top_corr_indices: torch.Tensor,  # [B, K', 2] -> (idxA, idxB)
        top_tar_embedding: torch.Tensor, # [B, N_A, D] -> E^A (unit norm)
        top_src_embedding: torch.Tensor, # [B, N_B, D] -> E^B (unit norm)
    ) -> torch.Tensor:
        B, Kp = top_corr_indices.shape[:2]
        batch_idx = torch.arange(B).unsqueeze(-1).expand(B, Kp)

        # Gather matched, already-normalized embeddings
        EA = top_tar_embedding[batch_idx, top_corr_indices[:, :, 0]]  # [B, K', D]
        EB = top_src_embedding[batch_idx, top_corr_indices[:, :, 1]]  # [B, K', D]
        EA = EA / (EA.norm(dim=-1, keepdim=True) + 1e-8)
        EB = EB / (EB.norm(dim=-1, keepdim=True) + 1e-8)
        w  = top_corr_values                                          # [B, K']

        dot_per_m = (EA * EB).sum(dim=-1)                             # [B, K']
        term1 = (w * dot_per_m).sum(dim=-1)                      # [B]
        # return  1- term1 
        return  term1 
    
    def forward(self, outputs):
        top_corr_values = outputs['top_corr_values']
        top_corr_indices = outputs['top_corr_indices']
        tar_embeddings = outputs['top_tar_embedding']
        src_embeddings = outputs['top_src_embedding']
        embedding_cov = self._embedding_cov_term(
            top_corr_values,
            top_corr_indices,
            tar_embeddings,
            src_embeddings,
        )
        # loss = embedding_cov.mean()
        return embedding_cov.mean()
    

class WeightEntropyLoss(nn.Module):
    def __init__(self):
        super(WeightEntropyLoss, self).__init__()

    def _embedding_entropy_term(
        self,
        top_corr_values: torch.Tensor,  # [B, K'] -> weights w_m
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        Entropy regularizer:
            λ_gap * exp( - Σ_m w_m log w_m )

        Args:
            top_corr_values: [B, K'] nonnegative weights (w_m).
            eps: small constant for numerical stability (avoids log(0)).

        Returns:
            Tensor of shape [B] with the entropy term per batch element.
        """
        w = top_corr_values.clamp_min(eps)  # [B, K']
        N = top_corr_values.shape[-1]

        H = -(w * torch.log(w)).sum(dim=-1) / torch.log(torch.tensor(N)) 
        return H

    def forward(self, outputs):
        top_corr_values = outputs['top_corr_values']
        entropy_term = self._embedding_entropy_term(
            top_corr_values,
        )
        return entropy_term.mean()
    
