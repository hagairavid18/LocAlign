import math
import torch
from torch import nn

from models.utils.math import compute_rmsd_torch


class QualityLoss(nn.Module):
    def __init__(self, weight_dict: dict[str, float], embedding_method: str = 'contrastive_clip_normalized'):
        super(QualityLoss, self).__init__()
        self._weight_dict = weight_dict
        self._embedding_term = EmbeddingSimilarityLoss(method=embedding_method)
        self._weight_entropy_term = WeightEntropyLoss()
        self._radius_gyration_term = RadiusGyrationLoss()
        self.register_buffer("corr_lambda", torch.tensor(weight_dict['corr_rmsd']) / 10)
        self._rmsd0 = 10.0

    def forward(self, outputs, epoch: int = 1):
        loss_dict = {}
        loss_dict_per_sample = {}
        corr_rmsd = outputs['corr_rmsd']
        embedding_similarity = self._embedding_term(outputs)
        gap = self._weight_entropy_term(outputs)
        radius_term = self._radius_gyration_term(outputs)
        # print all losses
        loss_dict_per_sample.update({
            'embedding': embedding_similarity,
            'gap': gap,
            'corr_rmsd': corr_rmsd,
            'radius': radius_term,
        })
       
        non_linear_corr_rmsd = corr_rmsd / (1 + corr_rmsd / self._rmsd0)
        loss = - self._weight_dict['embedding'] * embedding_similarity.mean() - self._weight_dict['gap'] * gap.mean() + self.corr_lambda * non_linear_corr_rmsd.mean() + self._weight_dict['radius'] * radius_term.mean()
        loss_dict.update({
            'embedding': embedding_similarity.mean(),
            'gap': gap.mean(),
            'corr_rmsd': corr_rmsd.mean(),
            'radius': radius_term.mean(),
        })
        loss_dict['per_sample'] = loss_dict_per_sample
        loss_dict['quality'] = loss
        return loss, loss_dict

class LocAlignLoss(nn.Module):
    def __init__(self, weight_dict: dict[str, float], return_non_linear: bool = True, embedding_method: list[str] = 'average_dot'):
        super(LocAlignLoss, self).__init__()
        self._quality_loss = QualityLoss(weight_dict, embedding_method=embedding_method)
        self._min_corr_lambda = weight_dict.get('corr_rmsd', 1.0) / 5.0
        self._max_corr_lambda = weight_dict.get('corr_rmsd', 1.0)

        self._ligand_loss = LigandLoss(return_non_linear=return_non_linear)
        self._ligand_loss_weight = weight_dict.get('ligand_rmsd', 1.0)
        # self._reduce = reduce

    def forward(self, batch, outputs, inference: bool = False, per_sample: bool = False, epoch: int = 1):
        rotation_ab_pred = outputs['pred_R']
        translation_ab_pred = outputs['pred_t']
        quality_loss, quality_loss_dict = self._quality_loss(outputs, epoch)
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

    def update_lambda(self, current_step: int, total_steps: int):
        # Update corr_lambda using a sine schedule
        t = min(current_step / total_steps, 1.0)
        new_value = self._min_corr_lambda + (self._max_corr_lambda - self._min_corr_lambda)  * math.sin(0.5 * math.pi * t)
        self._quality_loss.corr_lambda.fill_(new_value)


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
    def __init__(self,
                 method,
                 stop_gradient = False,
                 eps=1e-6
                 ):
        super(EmbeddingSimilarityLoss, self).__init__()
        assert method in ['average_dot',
                          'covariance',
                          'contrastive_diagonal',
                          'contrastive_axial',
                          'contrastive_axial_normalized',
                          'contrastive_reciprocal',
                          'contrastive_reciprocal_normalized',
                          'contrastive_clip',
                          'contrastive_clip_normalized',
                          ]
        self.method = method
        self.stop_gradient = stop_gradient
        self.eps = eps
        # self.beta = torch.nn.Parameter(torch.tensor(25.0))
        self.beta = torch.nn.Parameter(torch.tensor(55.0))
    
    def forward(self, outputs):
        top_corr_values = outputs['top_corr_values']
        tar_embeddings = outputs['corr_tar_embedding']
        src_embeddings = outputs['corr_src_embedding']
        
        if self.stop_gradient:
            top_corr_values = top_corr_values.detach()
        
        normalized_tar_embeddings = tar_embeddings / torch.sqrt( (tar_embeddings**2).sum(-1,keepdim=True) + self.eps)
        normalized_src_embeddings = src_embeddings / torch.sqrt( (src_embeddings**2).sum(-1,keepdim=True) + self.eps)
        
        if self.method == 'average_dot':
            dot_per_match =  (normalized_tar_embeddings * normalized_src_embeddings).sum(dim=-1)
            score = (top_corr_values * dot_per_match).sum(dim=-1)
            
        elif self.method == 'covariance':
            dot_per_match =  (normalized_tar_embeddings * normalized_src_embeddings).sum(dim=-1)
            average_match = (top_corr_values * dot_per_match).sum(dim=-1)
            
            average_tar_embeddings = torch.einsum('bi,bij->bj', top_corr_values, normalized_tar_embeddings )
            average_src_embeddings = torch.einsum('bi,bij->bj', top_corr_values, normalized_src_embeddings )            
            match_average = (average_tar_embeddings * average_src_embeddings).sum(-1)
            score =  average_match - match_average
            
        elif 'contrastive' in self.method:                    
            # beta = torch.full([1],fill_value=25.,device=top_corr_values.device)        
            all_dot_products = torch.einsum('bik,bjk->bij',normalized_tar_embeddings,normalized_src_embeddings)                
        
            if self.method in ['contrastive_diagonal','contrastive_diagonal_normalized']:                
                weights_positive_pairs = torch.diag_embed(top_corr_values)
                weights_negative_pairs = torch.unsqueeze(top_corr_values,axis=-1) * torch.unsqueeze(top_corr_values,axis=-2)
                
                all_dot_products_ = all_dot_products - all_dot_products.max(-1,keepdim=True).values.max(-2,keepdim=True).values # Not needed, maximum is 1
                distribution = (weights_positive_pairs+weights_negative_pairs) * torch.exp( self.beta *all_dot_products_)
                distribution = distribution / torch.clamp(torch.sum(distribution,axis=(-1,-2),keepdim=True), self.eps )        
                
                if self.method == 'contrastive_diagonal':                
                    score = torch.diagonal(distribution,dim1=1,dim2=2).sum(-1)
                    
                elif self.method == 'contrastive_diagonal_normalized':
                    unnormalized_score = torch.diagonal(distribution,dim1=1,dim2=2).sum(-1)
                    '''Before normalization:
                    - Equals to 0.5 * (1 + sum_i top_corr_values_i^2) if beta=0 and/or all embedding cosine similarities are equal (uninformative embeddings).
                    - Equals to 1 if embeddings are perfect (<E_i,E_i> >> <E_i, E_j> for all j!=i)
                    - Can be 0 if embeddings are worse than uninformative.
                    '''                    
                    baseline_score = 0.5 + 0.5 * (top_corr_values**2 ).sum(-1)
                    maximal_score = 1.
                    score = (unnormalized_score - baseline_score) / (maximal_score-baseline_score + self.eps)
                    '''
                    After normalization:
                    - Equals 0 if beta=0 and/or embeddings are uninformative
                    - Equals 1 if embeddings are perfect, irrespective of the number of correspondences.
                    - Can be negative if embeddings are worse than uninformative.                    
                    '''                                        
                else:
                    raise ValueError
            
            elif self.method in  [
                            'contrastive_axial',
                            'contrastive_axial_normalized',
                            'contrastive_reciprocal',
                            'contrastive_reciprocal_normalized',
                            'contrastive_clip',
                            'contrastive_clip_normalized']:
                
                                # Column-wise softmax (per-row distributions over j)
                distribution_col = top_corr_values.unsqueeze(-2) * torch.exp(
                    self.beta * (all_dot_products - all_dot_products.max(-1, keepdim=True).values)
                )
                distribution_col = distribution_col / torch.clamp(distribution_col.sum(axis=-1, keepdim=True), self.eps )
                # now distribution_col: (B, N, N), rows i normalized over j

                # Row-wise softmax (per-column distributions over i)
                distribution_row = top_corr_values.unsqueeze(-1) * torch.exp(
                    self.beta * (all_dot_products - all_dot_products.max(-2, keepdim=True).values)
                )
                distribution_row = distribution_row / torch.clamp(distribution_row.sum(axis=-2, keepdim=True), self.eps )
                # distribution_row: (B, N, N), columns j normalized over i

                
                if self.method == 'contrastive_axial':
                    distribution = 0.5 * (distribution_col + distribution_row)
                    score =  ( top_corr_values * torch.diagonal(distribution,dim1=1,dim2=2)  ).sum(-1)
                    
                elif self.method == 'contrastive_axial_normalized':
                    distribution = 0.5 * (distribution_col + distribution_row)
                    unnormalized_score =  ( top_corr_values * torch.diagonal(distribution,dim1=1,dim2=2)  ).sum(-1)
                    '''Before normalization:
                    - Equals to (sum_i top_corr_values_i^2) if beta=0 and/or all embedding cosine similarities are equal (uninformative embeddings).
                    - Equals to 1 if embeddings are perfect (<E_i,E_i> >> <E_i, E_j> for all j!=i)
                    - Can be 0 if embeddings are worse than uninformative.
                    '''                    
                    baseline_score = (top_corr_values**2 ).sum(-1)
                    maximal_score = 1.
                    # score = (unnormalized_score - baseline_score) / (maximal_score-baseline_score + self.eps)
                    score = (unnormalized_score - baseline_score) 
                    '''
                    After normalization:
                    - Equals 0 if beta=0 and/or embeddings are uninformative
                    - Equals 1 if embeddings are perfect, irrespective of the number of correspondences.
                    - Can be negative if embeddings are worse than uninformative.                    
                    '''
                                        
                elif self.method == 'contrastive_reciprocal':
                    distribution = distribution_col * distribution_row
                    score =  ( top_corr_values * torch.diagonal(distribution,dim1=1,dim2=2)  ).sum(-1)
                    
                elif self.method == 'contrastive_reciprocal_normalized':
                    distribution = distribution_col * distribution_row
                    unnormalized_score =  ( top_corr_values * torch.diagonal(distribution,dim1=1,dim2=2)  ).sum(-1) # Equals to sum( w_i^3) for beta=0 and/or uninformative embeddings 
                    '''Before normalization:
                    - Equals to (sum_i top_corr_values_i^3) if beta=0 and/or all embedding cosine similarities are equal (uninformative embeddings).
                    - Equals to 1 if embeddings are perfect (<E_i,E_i> >> <E_i, E_j> for all j!=i)
                    - Can be 0 if embeddings are worse than uninformative.
                    '''
                    
                    baseline_score = (top_corr_values**3 ).sum(-1)
                    maximal_score = 1.
                    # score = (unnormalized_score - baseline_score) / (maximal_score-baseline_score + self.eps)
                    score = (unnormalized_score - baseline_score) 
                    '''
                    After normalization:
                    - Equals 0 if beta=0 and/or embeddings are uninformative
                    - Equals 1 if embeddings are perfect, irrespective of the number of correspondences.
                    - Can be negative if embeddings are worse than uninformative.                    
                    '''
                    
                elif self.method == 'contrastive_clip': # Needs learnable beta
                    log_distribution = 0.5 * (  torch.log( distribution_col + self.eps) + torch.log( distribution_row + self.eps) )
                    score = ( top_corr_values * torch.diagonal(log_distribution ,dim1=1,dim2=2)  ).sum(-1)
                    
                    
                elif self.method == 'contrastive_clip_normalized':  # Needs learnable beta                    
                    log_distribution = 0.5 * (  torch.log( distribution_col + self.eps) + torch.log( distribution_row + self.eps) )                    
                    unnormalized_score = ( top_corr_values * torch.diagonal(log_distribution ,dim1=1,dim2=2)  ).sum(-1)
                    '''Before normalization:
                    - Equals to - Entropy(top_corr_values) if beta=0 and/or all embedding cosine similarities are equal (uninformative embeddings).
                    - Equals to 0 if embeddings are perfect (<E_i,E_i> >> <E_i, E_j> for all j!=i)
                    - Can be negative if embeddings are worse than uninformative.
                    '''
                    
                    baseline_score = (top_corr_values * torch.log(top_corr_values+ self.eps) ).sum(-1) # NegEntropy(top_corr_values)
                    # maximal_score = 0.
                    # score = (unnormalized_score - baseline_score) / (maximal_score-baseline_score + self.eps)
                    score = (unnormalized_score - baseline_score)
                    '''
                    After normalization:
                    - Equals 0 if beta=0 and/or embeddings are uninformative
                    - Equals 1 if embeddings are perfect, irrespective of the number of correspondences.
                    - Can be negative if embeddings are worse than uninformative.                    
                    '''
                    
                else:
                    raise ValueError
            else:
                raise ValueError
        return score
    

class WeightEntropyLoss(nn.Module):
    def __init__(self):
        super(WeightEntropyLoss, self).__init__()

    def forward(self, outputs):
        top_corr_values = outputs['top_corr_values']
        N = top_corr_values.size(1)
        H = -(top_corr_values * torch.log(top_corr_values)).sum(dim=-1) / torch.log(torch.tensor(N))
        return H
    

class RadiusGyrationLoss(nn.Module):
    def __init__(self):
        self.eps = 1e-6        
        super(RadiusGyrationLoss, self).__init__()
    
    def forward(self, outputs):
        corr_src_coordinates = outputs['corr_src_coordinates']
        corr_tar_coordinates = outputs['corr_tar_coordinates']

        # center the coordinates
        corr_src_coordinates = corr_src_coordinates - corr_src_coordinates.mean(dim=1,keepdim=True)
        corr_tar_coordinates = corr_tar_coordinates - corr_tar_coordinates.mean(dim=1,keepdim=True)
        # corr_tar_coordinates = corr_tar_coordinates - mean_corr_tar_coordinates.unsqueeze(1)
        top_corr_values = outputs['top_corr_values']
        mean_corr_src_coordinates = torch.einsum('bij,bi->bj', corr_src_coordinates,top_corr_values)
        mean_corr_tar_coordinates = torch.einsum('bij,bi->bj', corr_tar_coordinates,top_corr_values)


        
        mean2_corr_src_coordinates = torch.einsum('bij,bi->bj', corr_src_coordinates**2,top_corr_values)
        mean2_corr_tar_coordinates = torch.einsum('bij,bi->bj', corr_tar_coordinates**2,top_corr_values)
        radius_gyration_corr_src = torch.sqrt(  (mean2_corr_src_coordinates - mean_corr_src_coordinates**2).sum(-1) + self.eps )
        radius_gyration_corr_tar = torch.sqrt(  (mean2_corr_tar_coordinates - mean_corr_tar_coordinates**2).sum(-1) + self.eps )
        radius_gyration_corr = 0.5 * (radius_gyration_corr_tar + radius_gyration_corr_src)
        Neff_atoms = torch.exp( (- torch.log( top_corr_values + self.eps ) * top_corr_values).sum(-1) ) # Effective number of atoms	
        reference_radius_gyration = 1.3 * Neff_atoms ** (0.4) # A typical scaling value to expect from a compact alignment.
        score = radius_gyration_corr / reference_radius_gyration # Values above 1 indicate spread out alignment. Typically between 0.7 and 3.

        return score