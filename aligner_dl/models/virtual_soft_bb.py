from typing import Any
from models.layers.blocks import EmbeddingBlock
from models.recycling import RecyclingModule, rescale_and_concat
import torch
torch.set_float32_matmul_precision('medium')  # or 'high'

from models.soft_bb_base import SoftBBBase
from models.utils import move_batch_to_device, build_object, compute_transformation_from_corr_and_coord
from models.utils.plots import plot_correspondences


class VirtualSoftBB(SoftBBBase):
    def __init__(
            self, 
            loss: dict[str, Any], 
            optimizer: dict[str, Any], 
            input_layer: dict[str, Any], 
            scalar_layer: dict, 
            denoiser: dict, 
            max_iter: int = 5, 
            n_iter_train: int = 2, 
            top_k: int = 1200,
            n_iter_recycling: int = 3,
            plot_dir: str | None = None
            ) -> None:
       
        super().__init__(loss=loss, optimizer=optimizer, max_iter=max_iter, n_iter_train=n_iter_train, plot_dir=plot_dir)
        self._input_block = build_object(input_layer, 'models.layers')
        self._linear_iter0 = EmbeddingBlock(
            input_dim=256,  # 256 is the embedding dimension
            output_dim=1,
            n_blocks=3,
            hidden_dim=128,
            dropout=0.0,
            bias=False,
            norm_in_last_layer=False
        )
        self._linear_iter1 = EmbeddingBlock(
            input_dim=1278,  # 256 is the embedding dimension, and we add the previous iteration's embedding
            output_dim=1,
            n_blocks=3,
            hidden_dim=128,             
            dropout=0.0,
            bias=False,
            norm_in_last_layer=False
        )

        self._denoiser = build_object(denoiser, 'models')
        self._recycling = RecyclingModule()
        self._top_k = top_k

        self._n_recycling_iterations = n_iter_recycling
        # self.automatic_optimization = False  # We will handle the optimization manually
    
    def _get_atom_importance(
            self, 
            embedding: torch.Tensor, 
            mask:torch.Tensor,
            i
            ) -> torch.Tensor:
        if i == 0:
            return self._linear_iter0(embedding, mask=mask).squeeze(-1)
        else:
            # For the first iteration, we use the original embedding
            return self._linear_iter1(embedding, mask=mask).squeeze(-1)
    
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
        input_dtype = batch['tar_embedding'].dtype

        # #### Top K on atoms version #1
        orig_tar_embedding, orig_src_embedding  = self._input_block(batch['tar_embedding'], mask=batch['tar_mask']).to(input_dtype), self._input_block(batch['src_embedding'], mask =batch['src_mask']).to(input_dtype)
        
        optimal_transformations = []
        for i in range(self._n_recycling_iterations +1):
            if i == 0:
                tar_embedding, src_embedding = orig_tar_embedding, orig_src_embedding
            tar_atom_importance = self._get_atom_importance(tar_embedding, batch['tar_mask'], i)
            src_atom_importance = self._get_atom_importance(src_embedding, batch['src_mask'], i)

            top_tar_indices, top_tar_values = self._get_rectified_top_k(tar_atom_importance, batch['tar_mask'])
            top_src_indices, top_src_values = self._get_rectified_top_k(src_atom_importance, batch['src_mask'])

            # Now gather        
            top_src_embedding = src_embedding.gather(1, top_src_indices.unsqueeze(-1).expand(-1, -1, 256))
            top_tar_embedding = tar_embedding.gather(1, top_tar_indices.unsqueeze(-1).expand(-1, -1, 256))
            
            top_src_frames = batch['src_frames'].gather(1, top_src_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 3))
            top_tar_frames = batch['tar_frames'].gather(1, top_tar_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 3))
            
            top_src_mask = batch['src_mask'].gather(1, top_src_indices)
            top_tar_mask = batch['tar_mask'].gather(1, top_tar_indices)

            src_frames, tar_frames = top_src_frames, top_tar_frames 
            # if i == 0:
            #     top_src_embedding = top_src_embedding_orig
            #     top_tar_embedding = top_tar_embedding_orig
            soft_correspondences = self._get_soft_correspondences(top_src_embedding, top_tar_embedding, top_src_values, top_tar_values, top_src_mask, top_tar_mask).to(input_dtype)  
            
            top_corr_values, top_corr_indices = self._denoiser(soft_correspondences, src_frames, tar_frames)
            # plot_correspondences(batch,src_frames[:, :, 0, :], tar_frames[:, :, 0, :], top_corr_values, top_corr_indices)

            # Gather coordinates
            batch_indices = torch.arange(top_corr_indices.shape[0], device=top_corr_indices.device).unsqueeze(-1).expand(-1, top_corr_indices.shape[1])
            gathered_coord_tar = top_tar_frames[:, :, 0, :][batch_indices, top_corr_indices[:, :, 0]]  # [B, K, 3]
            gathered_coord_src = top_src_frames[:, :, 0, :][batch_indices, top_corr_indices[:, :, 1]]  # [B, K, 3]
            top_corr_residue_idx_tar = batch['tar_residue_indices'].gather(1, top_tar_indices)[batch_indices, top_corr_indices[:, :, 0]]
            top_corr_residue_idx_src = batch['src_residue_indices'].gather(1, top_src_indices)[batch_indices, top_corr_indices[:, :, 1]]
            corr_residue_indices = torch.stack([top_corr_residue_idx_src, top_corr_residue_idx_tar], dim=-1)  # [B, K, 2]

            optimal_transformation: dict[str, torch.Tensor] = compute_transformation_from_corr_and_coord(batch['max_length'], top_corr_values, gathered_coord_src, gathered_coord_tar, iter_limit=self._max_iter if not self.training else self._n_iter_train)
            optimal_transformations.append(optimal_transformation)
            transformed_src_coord = torch.matmul(batch['src_frames'][:, :, 0, :], optimal_transformation['pred_R']) + optimal_transformation['pred_t'][:,:3].unsqueeze(1)
            if i < self._n_recycling_iterations - 1:
                recycled_tar_embedding, recycled_src_embedding = self._recycling(transformed_src_coord, batch['tar_frames'][:, :, 0, :])
                # src_embedding = rescale_and_concat(orig_src_embedding, recycled_src_embedding)
                # tar_embedding = rescale_and_concat(orig_tar_embedding, recycled_tar_embedding)
                src_embedding = torch.cat([orig_src_embedding, recycled_src_embedding], dim=-1)
                tar_embedding = torch.cat([orig_tar_embedding, recycled_tar_embedding], dim=-1)
            
        if return_correspondences:
            return optimal_transformations, top_corr_values, corr_residue_indices
        return optimal_transformations

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = move_batch_to_device(batch, self.device)
        # print((batch['metadata'][0]['Ligand_ID'], batch['metadata'][0]['mov_protein'], batch['metadata'][0]['ref_protein'], batch['metadata'][0]['idx']))
        transformation_dicts = self._compute_soft_bb_algorithm(batch)
        loss = torch.tensor(0.0, device=self.device, dtype=batch['tar_embedding'].dtype)
        for transformation_dict in transformation_dicts:
            curr_loss, loss_dict = self._compute_loss(batch, transformation_dict['pred_R'], transformation_dict['pred_t'])
            loss += curr_loss
        loss /= len(transformation_dicts)  # Average loss over all iterations
        
        outputs = {'loss': loss , 'loss_dict': loss_dict, 'transformation_dict' :transformation_dict}    
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
        transformation_dicts = self._compute_soft_bb_algorithm(batch)
        loss = torch.tensor(0.0, device=self.device, dtype=batch['tar_embedding'].dtype)
        for transformation_dict in transformation_dicts:
            curr_loss, loss_dict = self._compute_loss(batch, transformation_dict['pred_R'], transformation_dict['pred_t'])
            loss += curr_loss
        loss /= len(transformation_dicts)  # Average loss over all iterations
        if self._plot:
            loss_iter1, _ = self._compute_loss(batch, transformation_dict['all_R'][0].detach(), transformation_dict['all_t'][0].detach())
            plot_correspondences(transformation_dict['all_gamma'], mask, batch['metadata'], [loss_iter1, loss], self._plot_dir)
            print(f"Loss: {loss.item()}")            
            metadata = batch['metadata']
            name = f"{metadata[0]['Ligand_ID']}_{metadata[0]['mov_protein']}_{metadata[0]['ref_protein']}_{metadata[0]['cath_degree']}"
            env = {
                'name':name,
                'batch':batch,
                'transformation_dict':transformation_dict,
                'mask':mask,
                'metadata':batch['metadata'],
                'loss_dict':loss_dict,
                'virtual_src_coord':virtual_src_coord,
                'virtual_tar_coord':virtual_tar_coord,
            }
            pickle.dump(env, open(os.path.join(self._plot_dir, f'all_info_{name}.pkl'),'wb') )
        
        outputs = {'loss': loss , 'loss_dict': loss_dict, 'transformation_dict': transformation_dict}
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
        transformation_dicts, corr_values, corr_residue_indices = self._compute_soft_bb_algorithm(batch, return_correspondences=True)
        
        outputs = {'transformation_dict': transformation_dicts[-1], 'metadata': batch['metadata'], 'corr_values': corr_values, 'corr_indices': corr_residue_indices}
        return outputs
