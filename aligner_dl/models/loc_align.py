from typing import Any
from models.recycling import RecyclingModule, rescale_and_concat
import torch
torch.set_float32_matmul_precision('medium')  # or 'high'

from models.soft_bb_base import SoftBBBase
from models.utils import move_batch_to_device, build_object, compute_transformation_from_corr_and_coord


class LocAlign(SoftBBBase):
    def __init__(
        self, 
        loss: dict[str, Any], 
        optimizer: dict[str, Any], 
        input_layer: dict[str, Any], 
        keypoints_selection: dict[str, Any], 
        denoiser: dict, 
        n_iter_recycling: int = 3
        ) -> None:
       
        super().__init__(loss=loss, optimizer=optimizer)
        self._input_block = build_object(input_layer, 'models.layers')
        self._keypoints_selection = build_object(keypoints_selection, 'models')
        self._denoiser = build_object(denoiser, 'models')
        self._recycling = RecyclingModule(recycle_scalar=True)

        self._n_recycling_iterations = n_iter_recycling
        self._corr_dropout = torch.nn.Dropout(p=0.1)
        # self.automatic_optimization = False  # We will handle the optimization manually
    
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
        scale = torch.sqrt(torch.tensor(dim))        
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
        
    def _run_step(
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
        
        all_iter_outputs = []
        recycled_tar_importance = None
        recycled_src_importance = None
        recycled_tar_embedding = None
        recycled_src_embedding = None
        # src_atom_importance = self._get_atom_importance(src_embedding, batch['src_mask'])
        # tar_atom_importance = self._get_atom_importance(tar_embedding, batch['tar_mask'])

        # topk_src_indices, topk_src_values = self._get_rectified_top_k(src_atom_importance, batch['src_mask'])
        # topk_tar_indices, topk_tar_values = self._get_rectified_top_k(tar_atom_importance, batch['tar_mask'])
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
            soft_correspondences = self._corr_dropout(soft_correspondences)
            
            top_corr_values, top_corr_indices, _ = self._denoiser(soft_correspondences, src_frames, tar_frames)

            # Gather coordinates
            batch_indices = torch.arange(top_corr_indices.shape[0]).unsqueeze(-1).expand(-1, top_corr_indices.shape[1])
            gathered_coord_src = keypoints_src_frames[:, :, 0, :][batch_indices, top_corr_indices[:, :, 1]]  # [B, K', 3]
            gathered_coord_tar = keypoints_tar_frames[:, :, 0, :][batch_indices, top_corr_indices[:, :, 0]]  # [B, K', 3]

            step_outputs: dict[str, torch.Tensor] = compute_transformation_from_corr_and_coord(top_corr_values, gathered_coord_src, gathered_coord_tar)
            
            # step_results['embedding_similarity'] = self._embedding_cov_term(top_corr_values, top_corr_indices, top_tar_embedding, top_src_embedding)
            # # step_results['embedding_similarity'] = self._embedding_cov_term(top_corr_values, top_corr_indices, batch['tar_pretrained_embeddings'].gather(1, topk_src_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)), batch['src_pretrained_embeddings'].gather(1, topk_src_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)))
            # step_results['gap'] = self._embedding_entropy_term(top_corr_values)
            step_outputs['top_corr_values'] = top_corr_values
            step_outputs['top_corr_indices'] = top_corr_indices
            step_outputs['top_tar_embedding'] = top_tar_embedding
            step_outputs['top_src_embedding'] = top_src_embedding
            all_iter_outputs.append(step_outputs)
            
            if i < self._n_recycling_iterations:
                transformed_src_coord = torch.matmul(batch['src_frames'][:, :, 0, :], step_outputs['pred_R']) + step_outputs['pred_t'][:,:3].unsqueeze(1)
                
                # retreive the original correspondences indices before top k selection
                original_topk_src_indices = topk_src_indices.gather(1, top_corr_indices[:, :, 1])
                original_topk_tar_indices = topk_tar_indices.gather(1, top_corr_indices[:, :, 0])
                original_top_corr_indices = torch.stack([original_topk_tar_indices, original_topk_src_indices], dim=-1)  # [B, K', 2]
                
                # recycled_tar_embedding, recycled_src_embedding = self._recycling(transformed_src_coord, batch['tar_frames'][:, :, 0, :], top_corr_values, original_top_corr_indices)      
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
            return all_iter_outputs, top_corr_values, corr_residue_indices, corr_atom_indices

        return all_iter_outputs

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = move_batch_to_device(batch, self.device)
        all_iter_outputs = self._run_step(batch)
        loss = torch.tensor(0.0, device=self.device, dtype=batch['tar_pretrained_embeddings'].dtype)
        for iter_outputs in all_iter_outputs:
            curr_loss, loss_dict = self._loss(batch, iter_outputs)
            loss += curr_loss
        loss /= len(all_iter_outputs)  # Average loss over all iterations

        outputs = {'loss': loss , 'loss_dict': loss_dict, 'transformation_dict': iter_outputs}
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
        all_iter_outputs = self._run_step(batch)
        loss = torch.tensor(0.0, device=self.device, dtype=batch['tar_pretrained_embeddings'].dtype)
        for iter_outputs in all_iter_outputs:
            curr_loss, loss_dict = self._loss(batch, iter_outputs)
            loss += curr_loss
        loss /= len(all_iter_outputs)  # Average loss over all iterations

        outputs = {'loss': loss , 'loss_dict': loss_dict, 'transformation_dict': iter_outputs}
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
        all_iter_results, corr_values, corr_residue_indices, corr_atom_indices = self._run_step(batch, return_correspondences=True)
        curr_loss, loss_dict = self._loss(batch, all_iter_results[-1], inference=True)
        
        # print(f"mean src {(torch.matmul(batch['src_frames'][:, :, 0, :], all_iter_results[-1]['pred_R']) + all_iter_results[-1]['pred_t']).mean()}")
        # print(f"mean tar {batch['tar_frames'][:, :, 0, :].mean()}")
        # Compute transformed source coordinates
        src_transformed = torch.matmul(batch['src_frames'][:, :, 0, :], all_iter_results[-1]['pred_R']) + all_iter_results[-1]['pred_t']
        tar = batch['tar_frames'][:, :, 0, :]

        # Masks -> expand to match xyz dimension
        src_mask = batch['src_mask'].unsqueeze(-1)  # [B, N, 1]
        tar_mask = batch['tar_mask'].unsqueeze(-1)

        # Apply masks before computing the mean
        src_mean = (src_transformed * src_mask).sum(1) / src_mask.sum(1).clamp(min=1)
        tar_mean = (tar * tar_mask).sum(1) / tar_mask.sum(1).clamp(min=1)

        print(f"mean src (masked): {src_mean}")
        print(f"mean tar (masked): {tar_mean}")

        outputs = {'transformation_dict': all_iter_results[-1], 'metadata': batch['metadata'], 'corr_values': corr_values, 'corr_indices': corr_residue_indices, 'corr_atom_indices': corr_atom_indices}
        outputs['loss'] = curr_loss
        outputs['loss_dict'] = loss_dict
        return outputs
