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
        denoiser: dict[str, Any],
        metric: dict[str, Any] = {},
        n_iter_recycling: int = 3
        ) -> None:

        super().__init__(loss=loss, optimizer=optimizer, metric=metric)
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
        soft_correspondences = self._corr_dropout(soft_correspondences)
        return soft_correspondences
        
    def _gather_keypoint_data(
        self,
        embedding: torch.Tensor,
        frames: torch.Tensor,
        mask: torch.Tensor,
        topk_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather embeddings, frames, and masks for selected keypoints.
        
        Args:
            embedding: Embedding tensor [B, N, D]
            frames: Frame tensor [B, N, 4, 3]
            mask: Mask tensor [B, N]
            topk_indices: Selected keypoint indices [B, K]
            
        Returns:
            Tuple of (keypoint_embeddings, keypoint_frames, keypoint_masks)
        """
        hidden_dim = embedding.shape[-1]
        keypoint_embedding = embedding.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, hidden_dim))
        keypoint_frames = frames.gather(1, topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 3))
        keypoint_mask = mask.gather(1, topk_indices)
        return keypoint_embedding, keypoint_frames, keypoint_mask
    
    def _prepare_embeddings_for_iteration(
        self,
        keypoint_embedding: torch.Tensor,
        recycled_embedding: torch.Tensor | None,
        topk_indices: torch.Tensor,
        is_first_iteration: bool
    ) -> torch.Tensor:
        """Prepare embeddings by optionally concatenating with recycled embeddings.
        
        Args:
            keypoint_embedding: Current keypoint embeddings [B, K, D]
            recycled_embedding: Recycled embeddings from previous iteration [B, N, D_recycled]
            topk_indices: Keypoint indices [B, K]
            is_first_iteration: Whether this is the first iteration
            
        Returns:
            Prepared embeddings [B, K, D'] where D' may be D or D+D_recycled
        """
        if is_first_iteration:
            return keypoint_embedding
        
        recycled_dim = recycled_embedding.shape[-1]
        gathered_recycled = recycled_embedding.gather(
            1, topk_indices.unsqueeze(-1).expand(-1, -1, recycled_dim)
        )
        return rescale_and_concat(keypoint_embedding, gathered_recycled)
    
    def _extract_correspondence_coordinates(
        self,
        keypoint_frames: torch.Tensor,
        top_corr_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract source and target coordinates for top correspondences.
        
        Args:
            keypoint_frames: Frames for keypoints [B, K, 4, 3]
            top_corr_indices: Correspondence indices [B, K', 2] where [:,:,0]=tar, [:,:,1]=src
            
        Returns:
            Tuple of (batch_indices, src_coords, tar_coords)
        """
        batch_size, num_corr = top_corr_indices.shape[:2]
        batch_indices = torch.arange(batch_size, device=top_corr_indices.device).unsqueeze(-1).expand(-1, num_corr)
        
        # Extract coordinates (first frame origin point at index 0)
        src_coords = keypoint_frames[0][:, :, 0, :][batch_indices, top_corr_indices[:, :, 1]]
        tar_coords = keypoint_frames[1][:, :, 0, :][batch_indices, top_corr_indices[:, :, 0]]
        
        return batch_indices, src_coords, tar_coords
    
    def _compute_recycling_inputs(
        self,
        batch: dict[str, torch.Tensor],
        step_outputs: dict[str, torch.Tensor],
        topk_src_indices: torch.Tensor,
        topk_tar_indices: torch.Tensor,
        top_corr_indices: torch.Tensor,
        top_corr_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute inputs for recycling module.
        
        Args:
            batch: Input batch
            step_outputs: Outputs from current step containing pred_R and pred_t
            topk_src_indices: Source keypoint indices
            topk_tar_indices: Target keypoint indices
            top_corr_indices: Correspondence indices
            top_corr_values: Correspondence values
            
        Returns:
            Tuple of (recycled_tar_emb, recycled_src_emb, recycled_tar_importance, recycled_src_importance)
        """
        # Transform source coordinates
        transformed_src_coord = torch.matmul(batch['src_frames'][:, :, 0, :], step_outputs['pred_R']) + step_outputs['pred_t'][:, :3].unsqueeze(1)

        # Map correspondence indices back to original space
        original_src_indices = topk_src_indices.gather(1, top_corr_indices[:, :, 1])
        original_tar_indices = topk_tar_indices.gather(1, top_corr_indices[:, :, 0])
        original_corr_indices = torch.stack([original_tar_indices, original_src_indices], dim=-1)

        return self._recycling(transformed_src_coord, batch['tar_frames'][:, :, 0, :],  top_corr_values, original_corr_indices)
    
    def _extract_correspondence_metadata(
        self,
        batch: dict[str, torch.Tensor],
        topk_src_indices: torch.Tensor,
        topk_tar_indices: torch.Tensor,
        top_corr_indices: torch.Tensor,
        batch_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract residue and atom indices for correspondences.
        
        Args:
            batch: Input batch containing residue and atom indices
            topk_src_indices: Source keypoint indices
            topk_tar_indices: Target keypoint indices
            top_corr_indices: Correspondence indices
            batch_indices: Batch dimension indices for gathering
            
        Returns:
            Tuple of (correspondence_residue_indices, correspondence_atom_indices)
        """
        # Gather residue indices
        src_residue_idx = batch['src_residue_indices'].gather(1, topk_src_indices)[batch_indices, top_corr_indices[:, :, 1]]
        tar_residue_idx = batch['tar_residue_indices'].gather(1, topk_tar_indices)[batch_indices, top_corr_indices[:, :, 0]]
        corr_residue_indices = torch.stack([src_residue_idx, tar_residue_idx], dim=-1)
        
        # Gather atom indices
        src_atom_idx = batch['src_atom_original_indices'].gather(1, topk_src_indices)[batch_indices, top_corr_indices[:, :, 1]]
        tar_atom_idx = batch['tar_atom_original_indices'].gather(1, topk_tar_indices)[batch_indices, top_corr_indices[:, :, 0]]
        corr_atom_indices = torch.stack([src_atom_idx, tar_atom_idx], dim=-1)

        src_atom_type = batch['src_atom_types'].gather(1, topk_src_indices)[batch_indices, top_corr_indices[:, :, 1]]
        tar_atom_type = batch['tar_atom_types'].gather(1, topk_tar_indices)[batch_indices, top_corr_indices[:, :, 0]]
        corr_atom_types = torch.stack([src_atom_type, tar_atom_type], dim=-1)
        
        return corr_residue_indices, corr_atom_indices, corr_atom_types

    def _run_step(
        self, 
        batch: dict[torch.Tensor],
        return_correspondences: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        The main algorithm of the model. Computes the soft correspondence matrix and the optimal 
        transformation between two sets of embeddings.

        Args:
            batch: Dictionary containing the input tensors.
            return_correspondences: Whether to return the correspondences. Defaults to False.

        Returns:
            Dictionary containing the predicted transformation matrices and other intermediate results.
        """
        # Process pretrained embeddings
        src_embedding = self._input_block(batch['src_pretrained_embeddings'], mask=batch['src_mask'])
        tar_embedding = self._input_block(batch['tar_pretrained_embeddings'], mask=batch['tar_mask'])
        
        # Initialize recycling state
        recycling_state = {
            'tar_importance': None,
            'src_importance': None,
            'tar_embedding': None,
            'src_embedding': None,
        }
        
        # Initialize cache for keypoint selection
        cache = {
            'src_value_key_query': None,
            'src_local_scalar_edges': None,
            'tar_value_key_query': None,
            'tar_local_scalar_edges': None,
        }
        
        all_iter_outputs = []
        num_iterations = self._n_recycling_iterations + 1
        
        for iteration in range(num_iterations):
            is_first_iteration = (iteration == 0)
            should_recycle = (iteration < self._n_recycling_iterations)
            
            # Select keypoints
            # Use initial importance from batch on first iteration if available, otherwise use recycling state
            src_importance = batch.get('src_initial_importance') if is_first_iteration else recycling_state['src_importance']
        
            tar_importance =batch.get('tar_initial_importance') if is_first_iteration else recycling_state['tar_importance']
            
            topk_src_indices, topk_src_values, cache['src_value_key_query'], cache['src_local_scalar_edges'] = (
                self._keypoints_selection(
                    src_embedding, 
                    batch['src_frames'], 
                    batch['src_neighbors'], 
                    batch['src_mask'],
                    previous_importance=src_importance,
                    cached_value_key_query=cache['src_value_key_query'],
                    cached_local_scalar_edges=cache['src_local_scalar_edges']
                )
            )
            
            topk_tar_indices, topk_tar_values, cache['tar_value_key_query'], cache['tar_local_scalar_edges'] = (
                self._keypoints_selection(
                    tar_embedding,
                    batch['tar_frames'],
                    batch['tar_neighbors'],
                    batch['tar_mask'],
                    previous_importance=tar_importance,
                    cached_value_key_query=cache['tar_value_key_query'],
                    cached_local_scalar_edges=cache['tar_local_scalar_edges']
                )
            )
            
            # Gather keypoint data
            kp_src_emb, kp_src_frames, kp_src_mask = self._gather_keypoint_data(src_embedding, batch['src_frames'], batch['src_mask'], topk_src_indices)
            kp_tar_emb, kp_tar_frames, kp_tar_mask = self._gather_keypoint_data(tar_embedding, batch['tar_frames'], batch['tar_mask'], topk_tar_indices)
            
                        

            # Prepare embeddings (concatenate with recycled if not first iteration)
            kp_src_emb_concated = self._prepare_embeddings_for_iteration(kp_src_emb, recycling_state['src_embedding'], topk_src_indices, is_first_iteration)
            kp_tar_emb_concated = self._prepare_embeddings_for_iteration(kp_tar_emb, recycling_state['tar_embedding'], topk_tar_indices, is_first_iteration)
            
            soft_correspondences = self._get_soft_correspondences(kp_src_emb_concated, kp_tar_emb_concated, topk_src_values, topk_tar_values, kp_src_mask, kp_tar_mask)
            
            top_corr_values, top_corr_indices, _ = self._denoiser(soft_correspondences, kp_src_frames, kp_tar_frames)
            
            batch_indices, corr_src_coord, corr_tar_coord = self._extract_correspondence_coordinates((kp_src_frames, kp_tar_frames), top_corr_indices)

            step_outputs = compute_transformation_from_corr_and_coord(top_corr_values, corr_src_coord, corr_tar_coord)
            
            # Store additional outputs
            step_outputs.update({
                'top_corr_values': top_corr_values,
                'top_corr_indices': top_corr_indices,
                'corr_tar_embedding': kp_tar_emb[batch_indices, top_corr_indices[:, :, 0]],
                'corr_src_embedding': kp_src_emb[batch_indices, top_corr_indices[:, :, 1]],
                'corr_src_coordinates': corr_src_coord,
                'corr_tar_coordinates': corr_tar_coord,
            })
            all_iter_outputs.append(step_outputs)
            
            # Compute recycling inputs for next iteration
            if should_recycle:
                (recycling_state['tar_embedding'], 
                 recycling_state['src_embedding'], 
                 recycling_state['tar_importance'], 
                 recycling_state['src_importance']) = self._compute_recycling_inputs(
                    batch, step_outputs, topk_src_indices, topk_tar_indices, 
                    top_corr_indices, top_corr_values
                )
                if batch.get('tar_initial_importance'):
                    recycling_state['tar_importance'] += batch['tar_initial_importance']
                if batch.get('src_initial_importance'):
                    recycling_state['src_importance'] += batch['src_initial_importance']
        
        # Return with correspondence metadata if requested
        if return_correspondences:
            corr_residue_indices, corr_atom_indices, corr_atom_types = self._extract_correspondence_metadata(
                batch, topk_src_indices, topk_tar_indices, top_corr_indices, batch_indices
            )
            return all_iter_outputs, top_corr_values, corr_residue_indices, corr_atom_indices, corr_atom_types
        
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
        batch = move_batch_to_device(batch, self.device)
        all_iter_outputs, corr_values, corr_residue_indices, corr_atom_indices, corr_atom_types = self._run_step(batch, return_correspondences=True)
        loss = torch.tensor(0.0, device=self.device, dtype=batch['tar_pretrained_embeddings'].dtype)
        for iter_outputs in all_iter_outputs:
            curr_loss, loss_dict = self._loss(batch, iter_outputs, per_sample=True)
            loss += curr_loss
        loss /= len(all_iter_outputs)  # Average loss over all iterations

        outputs = {'loss': loss , 'loss_dict': loss_dict, 'transformation_dict': iter_outputs, 'metadata': batch['metadata'], 'corr_values': corr_values, 'corr_indices': corr_residue_indices, 'corr_atom_indices': corr_atom_indices, 'corr_atom_types': corr_atom_types}
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
        batch = move_batch_to_device(batch, self.device)
        all_iter_results, corr_values, corr_residue_indices, corr_atom_indices, corr_atom_types = self._run_step(batch, return_correspondences=True)
        curr_loss, loss_dict = self._loss(batch, all_iter_results[-1], inference=True)
        

        outputs = {'transformation_dict': all_iter_results[-1], 'metadata': batch['metadata'], 'corr_values': corr_values, 'corr_indices': corr_residue_indices, 'corr_atom_indices': corr_atom_indices}
        outputs['loss'] = curr_loss
        outputs['loss_dict'] = loss_dict
        return outputs
