from typing import Any
from models.recycling import RecyclingModule, rescale_and_concat
import torch
torch.set_float32_matmul_precision('medium')  # or 'high'

from models.soft_bb_base import SoftBBBase
from models.utils import move_batch_to_device, build_object, compute_transformation_from_corr_and_coord
from models.utils.plots import plot_correspondences


class LocAlign(SoftBBBase):
    def __init__(
        self, 
        loss: dict[str, Any], 
        optimizer: dict[str, Any], 
        input_layer: dict[str, Any], 
        keypoints_selection: dict, 
        denoiser: dict, 
        corr_rmsd_lambda: float = 0.2,
        embedding_cosine_lambda: float = 0.1,
        gap_lambda: float = 1.0,
        ligaud_rmsd_lambda: float = 1.0,
        n_iter_recycling: int = 3,
        plot_dir: str | None = None
        ) -> None:
       
        super().__init__(loss=loss, optimizer=optimizer, corr_rmsd_lambda=corr_rmsd_lambda, embedding_cosine_lambda=embedding_cosine_lambda, gap_lambda=gap_lambda)
        super().__init__(loss=loss, optimizer=optimizer, corr_rmsd_lambda=corr_rmsd_lambda, embedding_cosine_lambda=embedding_cosine_lambda, gap_lambda=gap_lambda, ligand_rmsd_lambda=ligaud_rmsd_lambda)
        self._input_block = build_object(input_layer, 'models.layers')
        self._denoiser = build_object(denoiser, 'models')
        self._recycling = RecyclingModule(recycle_scalar=True)
        self._keypoints_selection = build_object(keypoints_selection, 'models')

        self._n_recycling_iterations = n_iter_recycling
        # self.automatic_optimization = False  # We will handle the optimization manually

    def _embedding_cov_term(
        self,
        top_corr_values: torch.Tensor,   # [B, K'] -> weights w_m
        top_corr_indices: torch.Tensor,  # [B, K', 2] -> (idxA, idxB)
        top_tar_embedding: torch.Tensor, # [B, N_A, D] -> E^A (unit norm)
        top_src_embedding: torch.Tensor, # [B, N_B, D] -> E^B (unit norm)
        lambda_emb: float = 1.0,
    ) -> torch.Tensor:
        B, Kp = top_corr_indices.shape[:2]
        device = top_corr_indices.device
        batch_idx = torch.arange(B, device=device).unsqueeze(-1).expand(B, Kp)

        # Gather matched, already-normalized embeddings
        EA = top_tar_embedding[batch_idx, top_corr_indices[:, :, 0]]  # [B, K', D]
        EB = top_src_embedding[batch_idx, top_corr_indices[:, :, 1]]  # [B, K', D]
        EA = EA / (EA.norm(dim=-1, keepdim=True) + 1e-8)
        EB = EB / (EB.norm(dim=-1, keepdim=True) + 1e-8)
        w  = top_corr_values                                          # [B, K']

        # term1 = (1/K') * Σ_m w_m (EA_m · EB_m)
        dot_per_m = (EA * EB).sum(dim=-1)                             # [B, K']
        term1 = (w * dot_per_m).sum(dim=-1)                      # [B]

        # μA = (1/K') Σ_m w_m EA_m ; μB similarly
        w_exp = w.unsqueeze(-1)                                       # [B, K', 1]
        muA = (w_exp * EA).sum(dim=1)                            # [B, D]
        muB = (w_exp * EB).sum(dim=1)                            # [B, D]

        term2 = (muA * muB).sum(dim=-1)                               # [B]

        # Final: -λ_emb [ term1 - term2 ]
        return  term1 - term2                          # [B]

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


        H = -(w * torch.log(w)).sum(dim=-1)  
        return torch.exp(H)      # [B]
        # return H
    
    def _get_atom_importance(
        self, 
        embedding: torch.Tensor, 
        mask:torch.Tensor
        ) -> torch.Tensor:
        return self._linear(embedding, mask=mask).squeeze(-1)
    
    def _get_rectified_top_k(
        self, 
        scalar_values: torch.Tensor, 
        mask: torch.Tensor
        ) -> tuple[torch.Tensor,torch.Tensor]:
        """
        Computes the top k scalar values and their indices, while ensuring that the mask is applied correctly.
        Args:
            scalar_values (torch.Tensor): Values to compute the top k from.
            mask (torch.Tensor): Mask to apply on the scalar values.

        Returns:
            tuple[torch.Tensor,torch.Tensor]: Top k indices and their corresponding scalar values.
        """        
        
        scalar_values = scalar_values.masked_fill(~mask, -float('inf'))
                
        top_k_plus_one_scalar, top_k_plus_one_indices = torch.topk(scalar_values, self._top_k + 1, dim=-1, largest=True) # I removed the train/eval condition for now...
        
        top_k_plus_one_mask = mask.gather(1, top_k_plus_one_indices) #
        top_k_plus_one_scalar.masked_fill_(~top_k_plus_one_mask, 0) # These two lines are to deal with the edge case where num_real_atoms < _top_k. In this case and without the fix, top_k_plus_one_scalar[:,:,_top_k] = -infty.
        
        top_k_scalar = top_k_plus_one_scalar[:, :self._top_k] - top_k_plus_one_scalar[:, self._top_k:]
        top_k_indices = top_k_plus_one_indices[:, :self._top_k]        
        return top_k_indices, top_k_scalar
    
    def _get_soft_correspondences(
        self,
        src_embedding: torch.Tensor,
        tar_embedding: torch.Tensor,
        src_scalar: torch.Tensor,
        tar_scalar: torch.Tensor,                                                      
        src_mask: torch.Tensor,
        tar_mask: torch.Tensor
        ) -> torch.Tensor: 
        '''
           Computes the soft correspondences between the source and target embeddings.
              The soft correspondences are computed by first computing the dot product between the source and target embeddings, then applying a softmax over the source and target embeddings.
        Args:
            src_embedding (torch.Tensor): Source embedding of shape [Batch Size, src_size, dim].
            tar_embedding (torch.Tensor): Target embedding of shape [Batch Size, tar_size, dim].
            src_scalar (torch.Tensor): Scalar values for the source embedding of shape [Batch Size, src_size].
            tar_scalar (torch.Tensor): Scalar values for the target embedding of shape [Batch Size, tar_size].
            src_mask (torch.Tensor): Mask for the source embedding of shape [Batch Size, src_size].
            tar_mask (torch.Tensor): Mask for the target embedding of shape [Batch Size, tar_size].
        Returns:
            torch.Tensor: Soft correspondences of shape [Batch Size, tar_size, src_size].  
        '''
        
        dim = src_embedding.shape[-1]
        scale = torch.sqrt(torch.tensor(dim, device=src_embedding.device, dtype=src_embedding.dtype))        
        dot_products = torch.bmm(tar_embedding, src_embedding.transpose(1, 2)) / scale
        
        # first softmax over src
        exp_dot_products = torch.exp(dot_products - dot_products.max(2, keepdims=True)[0])        
        softmax_over_src = exp_dot_products * (src_scalar * src_mask).unsqueeze(1)
        softmax_over_src /= torch.sum(softmax_over_src, dim=2, keepdim=True)
        
        # second softmax over tar
        exp_dot_products = torch.exp(dot_products - dot_products.max(1, keepdims=True)[0])
        softmax_over_tar = exp_dot_products * (tar_scalar * tar_mask).unsqueeze(2)
        softmax_over_tar /= torch.sum(softmax_over_tar, dim=1, keepdim=True)
        
        soft_correspondences = softmax_over_src * softmax_over_tar
        return soft_correspondences
        
    def _compute_soft_bb_algorithm(
        self, 
        batch: dict[torch.Tensor],
        return_correspondences: bool = False
        ) -> dict[str, torch.Tensor]:
        """
        The main algorithm of the model. Computes the soft correspondence matrix and the optimal transformation between two sets of embeddings.

        Args:
            batch (dict[torch.Tensor]): Dictionary containing the input tensors.
            return_correspondences (bool, optional): Whether to return the correspondences. Defaults to False.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing the predicted transformation matrices and other intermediate results.
        """

        # Process pretrained embeddings
        tar_embedding, src_embedding  = self._input_block(batch['tar_pretrained_embeddings'], mask=batch['tar_mask']), self._input_block(batch['src_pretrained_embeddings'], mask =batch['src_mask'])
        
        all_iter_results = []
        recycled_tar_importance = None
        recycled_src_importance = None
        recycled_tar_embedding = None
        recycled_src_embedding = None
        for i in range(self._n_recycling_iterations +1):
                
                
            topk_src_indices, topk_src_values = self._keypoints_selection(src_embedding, batch['src_frames'], batch['src_neighbors'], batch['src_mask'], previous_importance=recycled_src_importance)
            topk_tar_indices, topk_tar_values = self._keypoints_selection(tar_embedding, batch['tar_frames'], batch['tar_neighbors'], batch['tar_mask'], previous_importance=recycled_tar_importance)

            # Now gather
            hidden_dim = src_embedding.shape[-1]        
            keypoints_src_embedding = src_embedding.gather(1, topk_src_indices.unsqueeze(-1).expand(-1, -1, hidden_dim))
            keypoints_tar_embedding = tar_embedding.gather(1, topk_tar_indices.unsqueeze(-1).expand(-1, -1, hidden_dim))
            
            keypoints_src_frames = batch['src_frames'].gather(1, topk_src_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 3))
            keypoints_tar_frames = batch['tar_frames'].gather(1, topk_tar_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 3))
            
            keypoints_src_mask = batch['src_mask'].gather(1, topk_src_indices)
            keypoints_tar_mask = batch['tar_mask'].gather(1, topk_tar_indices)

            src_frames, tar_frames = keypoints_src_frames, keypoints_tar_frames 
            if i == 0:
                top_src_embedding = keypoints_src_embedding
                top_tar_embedding = keypoints_tar_embedding
            else:
                top_src_embedding = rescale_and_concat(keypoints_src_embedding, recycled_src_embedding.gather(1, topk_src_indices.unsqueeze(-1).expand(-1, -1, 1022)))
                top_tar_embedding = rescale_and_concat(keypoints_tar_embedding, recycled_tar_embedding.gather(1, topk_tar_indices.unsqueeze(-1).expand(-1, -1, 1022)))
            soft_correspondences = self._get_soft_correspondences(top_src_embedding, top_tar_embedding, topk_src_values, topk_tar_values, keypoints_src_mask, keypoints_tar_mask)
            
            top_corr_values, top_corr_indices, _ = self._denoiser(soft_correspondences, src_frames, tar_frames)

            # Gather coordinates
            batch_indices = torch.arange(top_corr_indices.shape[0], device=top_corr_indices.device).unsqueeze(-1).expand(-1, top_corr_indices.shape[1])
            gathered_coord_src = keypoints_src_frames[:, :, 0, :][batch_indices, top_corr_indices[:, :, 1]]  # [B, K', 3]
            gathered_coord_tar = keypoints_tar_frames[:, :, 0, :][batch_indices, top_corr_indices[:, :, 0]]  # [B, K', 3]

            step_results: dict[str, torch.Tensor] = compute_transformation_from_corr_and_coord(top_corr_values, gathered_coord_src, gathered_coord_tar)
            
            step_results['embedding_similarity'] = self._embedding_cov_term(top_corr_values, top_corr_indices, top_tar_embedding, top_src_embedding)
            step_results['gap'] = self._embedding_entropy_term(top_corr_values) / top_corr_values.shape[1]
            all_iter_results.append(step_results)
            
            if i < self._n_recycling_iterations:
                transformed_src_coord = torch.matmul(batch['src_frames'][:, :, 0, :], step_results['pred_R']) + step_results['pred_t'][:,:3].unsqueeze(1)
                
                # retreive the original correspondences indices before top k selection
                original_topk_src_indices = topk_src_indices.gather(1, top_corr_indices[:, :, 1])
                original_topk_tar_indices = topk_tar_indices.gather(1, top_corr_indices[:, :, 0])
                original_top_corr_indices = torch.stack([original_topk_tar_indices, original_topk_src_indices], dim=-1)  # [B, K', 2]
                
                recycled_tar_embedding, recycled_src_embedding, recycled_tar_importance, recycled_src_importance = self._recycling(transformed_src_coord, batch['tar_frames'][:, :, 0, :], top_corr_values, original_top_corr_indices)      
            
        if return_correspondences:
            top_corr_residue_idx_tar = batch['tar_residue_indices'].gather(1, topk_tar_indices)[batch_indices, top_corr_indices[:, :, 0]]
            top_corr_residue_idx_src = batch['src_residue_indices'].gather(1, topk_src_indices)[batch_indices, top_corr_indices[:, :, 1]]
            corr_residue_indices = torch.stack([top_corr_residue_idx_src, top_corr_residue_idx_tar], dim=-1)  # [B, K, 2]

            # find the original atom indices corresponding to the top correspondences. just take the indices of the atoms used for the correspondences
            # there is no src_atom_indices in the batch, only residue indices
            corr_atom_indices_tar = batch['tar_atom_original_indices'].gather(1, topk_tar_indices)[batch_indices, top_corr_indices[:, :, 0]]
            corr_atom_indices_src = batch['src_atom_original_indices'].gather(1, topk_src_indices)[batch_indices, top_corr_indices[:, :, 1]]
            corr_atom_indices = torch.stack([corr_atom_indices_src, corr_atom_indices_tar], dim=-1)  # [B, K, 2]
            return all_iter_results, top_corr_values, corr_residue_indices, corr_atom_indices
        
        return all_iter_results

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = move_batch_to_device(batch, self.device)
        all_iter_results = self._compute_soft_bb_algorithm(batch)
        loss = torch.tensor(0.0, device=self.device, dtype=batch['tar_pretrained_embeddings'].dtype)
        for iter_results in all_iter_results:
            curr_loss, loss_dict = self._compute_loss(batch, iter_results['pred_R'], iter_results['pred_t'],  iter_results['corr_rmsd'], iter_results['embedding_similarity'], iter_results['gap'])
            loss += curr_loss
        loss /= len(all_iter_results)  # Average loss over all iterations
        
        outputs = {'loss': loss , 'loss_dict': loss_dict, 'transformation_dict': iter_results}    
        return outputs

    def validation_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Validation step for the model. Computes the loss and the metrics for the model.

        Args:
            batch (dict[str, torch.Tensor]): 

        Returns:
            dict[str, torch.Tensor]: 
        """
        # print(f"tar protein: {batch['metadata'][0]['ref_protein']}{batch['metadata'][0]['ref_chain']} src protein: {batch['metadata'][0]['mov_protein']}{batch['metadata'][0]['mov_chain']}")
        batch = move_batch_to_device(batch, self.device)
        all_iter_results = self._compute_soft_bb_algorithm(batch)
        loss = torch.tensor(0.0, device=self.device, dtype=batch['tar_pretrained_embeddings'].dtype)
        for iter_results in all_iter_results:
            curr_loss, loss_dict = self._compute_loss(batch, iter_results['pred_R'], iter_results['pred_t'], iter_results['corr_rmsd'], iter_results['embedding_similarity'], iter_results['gap'])
            loss += curr_loss
        loss /= len(all_iter_results)  # Average loss over all iterations
        if self._plot:
            loss_iter1, _ = self._compute_loss(batch, iter_results['all_R'][0].detach(), iter_results['all_t'][0].detach())
            plot_correspondences(iter_results['all_gamma'], batch['metadata'], [loss_iter1, loss], self._plot_dir)
            print(f"Loss: {loss.item()}")            
        
        outputs = {'loss': loss , 'loss_dict': loss_dict, 'transformation_dict': iter_results}
        self._metrics.update(batch, outputs)
        return outputs
    
    def inference_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Validation step for the model. Computes the loss and the metrics for the model.

        Args:
            batch (dict[str, torch.Tensor]): 

        Returns:
            dict[str, torch.Tensor]: 
        """
        # print(f"tar protein: {batch['metadata'][0]['ref_protein']}{batch['metadata'][0]['ref_chain']} src protein: {batch['metadata'][0]['mov_protein']}{batch['metadata'][0]['mov_chain']}")
        batch = move_batch_to_device(batch, self.device)
        all_iter_results, corr_values, corr_residue_indices, corr_atom_indices = self._compute_soft_bb_algorithm(batch, return_correspondences=True)
        
        outputs = {'transformation_dict': all_iter_results[-1], 'metadata': batch['metadata'], 'corr_values': corr_values, 'corr_indices': corr_residue_indices, 'corr_atom_indices': corr_atom_indices}
        return outputs
