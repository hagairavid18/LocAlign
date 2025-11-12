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
        loss_dict_per_sample = {}
        corr_rmsd = outputs['corr_rmsd']
        embedding_similarity = self._embedding_term(outputs)
        gap = self._weight_entropy_term(outputs)
        loss_dict_per_sample.update({
            'embedding': embedding_similarity,
            'gap': gap,
            'corr_rmsd': corr_rmsd,
        })
        loss = - self._weight_dict['embedding'] * embedding_similarity.mean() - self._weight_dict['gap'] * gap.mean() + self._weight_dict['corr_rmsd'] * corr_rmsd.mean()
        loss_dict.update({
            'embedding': embedding_similarity.mean(),
            'gap': gap.mean(),
            'corr_rmsd': corr_rmsd.mean(),
        })
        loss_dict['per_sample'] = loss_dict_per_sample
        loss_dict['quality'] = loss
        return loss, loss_dict

class LocAlignLoss(nn.Module):
    def __init__(self, weight_dict: dict[str, float], return_non_linear: bool = True):
        super(LocAlignLoss, self).__init__()
        self._quality_loss = QualityLoss(weight_dict)

        self._ligand_loss = LigandLoss(return_non_linear=return_non_linear)
        self._ligand_loss_weight = weight_dict.get('ligand_rmsd', 1.0)
        # self._reduce = reduce

    def forward(self, batch, outputs, inference: bool = False, per_sample: bool = False):
        rotation_ab_pred = outputs['pred_R']
        translation_ab_pred = outputs['pred_t']
        quality_loss, quality_loss_dict = self._quality_loss(outputs)
        if inference:
            quality_loss_dict['loss'] = quality_loss
            return quality_loss, quality_loss_dict
        ligand_rmsd = self._ligand_loss(batch, rotation_ab_pred, translation_ab_pred, reduce=False)
        total_loss = quality_loss + self._ligand_loss_weight * ligand_rmsd.mean()
        # ligand_loss_dict = {'ligand_rmsd': ligand_rmsd}
        quality_loss_dict['per_sample']['ligand_rmsd'] = ligand_rmsd
        ligand_loss_dict = {'ligand_rmsd': ligand_rmsd.mean()}
        if not per_sample:
            quality_loss_dict.pop('per_sample', None)
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
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
    
    def forward(self, outputs):
        top_corr_values = outputs['top_corr_values']
        tar_embeddings = outputs['corr_tar_embedding']
        src_embeddings = outputs['corr_src_embedding']
        dot_per_m = self.cosine_similarity(tar_embeddings, src_embeddings)
        embedding_cov = (top_corr_values * dot_per_m).sum(dim=-1)
        return embedding_cov
    

class WeightEntropyLoss(nn.Module):
    def __init__(self):
        super(WeightEntropyLoss, self).__init__()

    def forward(self, outputs):
        top_corr_values = outputs['top_corr_values']
        N = top_corr_values.size(1)
        H = -(top_corr_values * torch.log(top_corr_values)).sum(dim=-1) / torch.log(torch.tensor(N))
        return H

