from typing import Any
import torch

from models.soft_bb_base import SoftBBBase
from models.utils import move_batch_to_device, build_object, compute_transformation_from_corr_and_coord, mask_and_normalize_matrix
from models.utils.math import group_lasso_regularization
from models.utils.plots import plot_correspondences, plot_offsets
torch.set_float32_matmul_precision('medium')


class VirtualSoftBB(SoftBBBase):
    def __init__(self, loss: dict[str, Any], optimizer: dict[str, Any], input_layer: dict[str, Any], virtual_layer: dict, scalar_layer: dict, denoiser: dict, max_iter: int = 5, n_iter_train: int = 2, top_k: int = 1200, plot_dir: str | None = None) -> None:
       
        super().__init__(loss=loss, optimizer=optimizer, max_iter=max_iter, n_iter_train=n_iter_train, plot_dir=plot_dir)
        self._input_block = build_object(input_layer, 'models.layers')
        self._linear = build_object(scalar_layer, 'models.layers')  # Projects tar_embedding to a scalar
        self._virtual_point_block = build_object(virtual_layer, 'models.layers')
        self._denoiser = build_object(denoiser, 'models')
        self._top_k = top_k
        # self._automatic_optimization = False
    
    def _get_scalar(self, embedding: torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        return self._linear(embedding, mask=mask).squeeze(-1)
    
    def _get_rectified_top_k(self,scalar:torch.Tensor, mask:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        
        scalar = scalar.masked_fill(mask, -float('inf'))
                
        top_k_plus_one_scalar, top_k_plus_one_indices = torch.topk(scalar, self._top_k + 1, dim=-1, largest=True) # I removed the train/eval condition for now...
        
        top_k_plus_one_mask = mask.gather(1, top_k_plus_one_indices) #
        top_k_plus_one_scalar.masked_fill(top_k_plus_one_mask,0) # These two lines are to deal with the edge case where num_real_atoms < _top_k. In this case and without the fix, top_k_plus_one_scalar[:,:,_top_k] = -infty.
        
        top_k_scalar = top_k_plus_one_scalar[:, :self._top_k] - top_k_plus_one_scalar[:, self._top_k:]
        top_k_indices = top_k_plus_one_indices[:,:self.k]        
        return top_k_indices,top_k_scalar
    
    def _get_soft_correspondences(self,
                                    src_embedding: torch.Tensor,
                                    tar_embedding: torch.Tensor,
                                    src_scalar: torch.Tensor,
                                    tar_scalar: torch.Tensor,                                                      
                                    src_mask: torch.Tensor,
                                    tar_mask: torch.Tensor) -> torch.Tensor: 
        '''
        These are the differences compared to your version:
        1/ Most importantly, we want that if an atom has scalar=0 or close to 0, it should not contribute at all. This guarantees differentiability when combined with the top-K.
        2/ I used dot products and softmax instead of the distance and softmin. Since there was a layernorm before, it should be exactly identical. dot product might also be faster.
        3/ There is a tiny fix that might help with numerical underflows...   
        4/ I did not use adaptive temperature... Maybe you will need to put it back.     
        '''
        
        dim = src_embedding.shape[-1]
        scale = torch.sqrt(torch.tensor(dim, device=src_embedding.device, dtype=src_embedding.dtype))        
        dot_products = torch.bmm(tar_embedding, src_embedding.transpose(1, 2)) / scale # [Batch Size , tar_size , src_size] Hopefully it's the right ordering...
        # exp_dot_products = torch.exp( dot_products - dot_products.max() )  ## This is what you used, but I did it separately for each dim to guarantee that there are no underflows.
        
        # first softmax over src
        exp_dot_products = torch.exp( dot_products - dot_products.max(2,keepdims=True) )        
        softmax_over_src = exp_dot_products * (src_scalar * src_mask).unsqueeze(1)
        softmax_over_src /= torch.sum(softmax_over_src, dim=2, keepdim=True)
        
        # second softmax over tar
        exp_dot_products = torch.exp( dot_products - dot_products.max(1,keepdims=True) )
        softmax_over_tar = exp_dot_products * (tar_scalar * tar_mask).unsqueeze(2)
        softmax_over_tar /= torch.sum(softmax_over_src, dim=1, keepdim=True)
        
        soft_correspondences = softmax_over_src * softmax_over_tar
        return soft_correspondences
                                                     
                
    def _get_top_k_indices(self, src_embedding: torch.Tensor, tar_embedding: torch.Tensor, src_mask: torch.Tensor, tar_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the top k indices of the source and target embeddings based on the distance matrix.

        Args:
            src_embedding (torch.Tensor): Source embedding.
            tar_embedding (torch.Tensor): Target embedding.
            src_mask (torch.Tensor): Source mask.
            tar_mask (torch.Tensor): Target mask.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Top k indices for source and target embeddings.
        """
        src_scalar = self._linear(src_embedding, mask=src_mask).squeeze(-1)
        tar_scalar = self._linear(tar_embedding, mask=tar_mask).squeeze(-1)
        src_scalar = src_scalar.masked_fill(~src_mask, -float('inf'))
        tar_scalar = tar_scalar.masked_fill(~tar_mask, -float('inf'))
        top_src_indices = torch.topk(src_scalar, k=self._top_k if self.training else src_scalar.shape[1], dim=-1).indices
        top_tar_indices = torch.topk(tar_scalar, k=self._top_k if self.training else src_scalar.shape[1], dim=-1).indices
        return top_src_indices, top_tar_indices
    
    def _get_distance_matrix(self, src_embedding: torch.Tensor, tar_embedding: torch.Tensor) -> torch.Tensor: 
        dim = src_embedding.shape[-1]
        scale = torch.sqrt(torch.tensor(dim, device=src_embedding.device, dtype=src_embedding.dtype))
        diff = src_embedding.unsqueeze(1) - tar_embedding.unsqueeze(2)
        distance_matrix = torch.sum(torch.mul(diff, diff), dim=-1) / scale

        return distance_matrix
    
    def _transform_offsets_with_frames(self, offsets: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        """
        Transforms the offsets with the frames of each residue.
        Each predicted offset is transformed with the corresponding frame of the residue, by applying rotation and translation.
        

        Args:
            offsets (torch.Tensor): relative 3D coordinates of points.
            frames (torch.Tensor): corresponding frames of the residues. 

        Returns:
            torch.Tensor: _description_
        """
        transformations = torch.zeros(frames.shape[0], frames.shape[1], 4, 4, device=frames.device)
        transformations[:, :, :3, :3] = frames[:,:, 1:4, :].transpose(2,3)
        transformations[:, :, :3, 3] = frames[:, :, 0, :] # TODO: wait for bugfix

        transformations[:, :, 3, 3] = 1
        offsets = torch.cat([offsets, torch.ones_like(offsets[:, :, :1])], dim=-1)
        return torch.matmul(transformations, offsets.unsqueeze(-1)).squeeze(-1)[..., :3]

    def _create_virtual_coordinates(self, embedding: torch.Tensor, frames: torch.Tensor, mask: torch.Tensor, metadata: str) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            embedding (torch.Tensor): embedding.
            frames (torch.Tensor): Frames of the residues.
            mask (torch.Tensor): Mask of the residues.
            metadata (str): Metadata of the residues.

        Returns:
            torch.Tensor: virtual points per input, transformed with the frames.
            torch.Tensor: the predicted offsets.
        """        
        offsets = self._virtual_point_block(embedding, mask)
        transformed_offsets = self._transform_offsets_with_frames(offsets, frames)
        if self._plot:
            plot_offsets(offsets[mask], frames[:, :, 0, :][mask], metadata, self._plot_dir)
        transformed_offsets  = transformed_offsets * mask.unsqueeze(-1)
        return transformed_offsets, offsets * mask.unsqueeze(-1)
        
    def _compute_soft_bb_algorithm(self, batch: dict[torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        The main algorithm of the model. Computes the soft correspondence matrix and the optimal transformation between two sets of embeddings.

        Args:
            batch (dict[torch.Tensor]): Dictionary containing the input tensors.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]: A tuple containing the transformation dictionary and the embeddings dictionary.
        """
        input_dtype = batch['tar_embedding'].dtype

        #### Top K on atoms version #1
        tar_embedding, src_embedding  = self._input_block(batch['tar_embedding'], mask=batch['tar_mask']).to(input_dtype), self._input_block(batch['src_embedding'], mask =batch['src_mask']).to(input_dtype)
        # tar_embedding, src_embedding  = batch['tar_embedding'], batch['src_embedding']                
        top_src_indices, top_tar_indices = self._get_top_k_indices(src_embedding=batch['src_embedding'], tar_embedding=batch['tar_embedding'], src_mask=batch['src_mask'], tar_mask=batch['tar_mask'])
        src_embedding = src_embedding.gather(1, top_src_indices.unsqueeze(-1).expand(-1, -1, 256))
        tar_embedding = tar_embedding.gather(1, top_tar_indices.unsqueeze(-1).expand(-1, -1, 256))
        distance_matrix = self._get_distance_matrix(src_embedding=src_embedding, tar_embedding=tar_embedding).to(input_dtype)

        # Now gather
        src_frames = batch['src_frames'].gather(1, top_src_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 3))
        tar_frames = batch['tar_frames'].gather(1, top_tar_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 3))
        src_mask = batch['src_mask'].gather(1, top_src_indices)
        tar_mask = batch['tar_mask'].gather(1, top_tar_indices)
        soft_correspondences = mask_and_normalize_matrix(distance_matrix, src_mask, tar_mask, src_embedding, tar_embedding).to(input_dtype)
        ### End of version #1

        '''
        #### Top K on atoms version #2
        tar_embedding, src_embedding = batch['tar_embedding'], batch['src_embedding']
        tar_mask, src_mask = batch['tar_mask'], batch['src_mask']
        tar_frames, src_frames = batch['tar_frames'], batch['src_frames']        
        tar_embedding = self._input_block(tar_embedding, mask=tar_mask).to(input_dtype)
        src_embedding = self._input_block(src_embedding, mask =src_mask).to(input_dtype)        
        tar_scalar = self._get_scalar(tar_embedding)
        src_scalar = self._get_scalar(src_embedding)
        
        top_tar_indices, top_tar_scalar = self._get_rectified_top_k( tar_scalar,tar_mask )
        top_src_indices, top_src_scalar = self._get_rectified_top_k( src_scalar,src_mask )

        # Now gather        
        top_src_embedding = src_embedding.gather(1, top_src_indices.unsqueeze(-1).expand(-1, -1, 256))
        top_tar_embedding = tar_embedding.gather(1, top_tar_indices.unsqueeze(-1).expand(-1, -1, 256))
        
        top_src_frames = src_frames.gather(1, top_src_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 3))
        top_tar_frames = tar_frames.gather(1, top_tar_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 3))
        
        top_src_mask = src_mask.gather(1, top_src_indices)
        top_tar_mask = tar_mask.gather(1, top_tar_indices)
        
                        
        soft_correspondences = self._get_soft_correspondences(
                                                     top_src_embedding,  top_tar_embedding,
                                                     top_src_scalar, top_tar_scalar,
                                                     top_src_mask, top_tar_mask, 
                                                     ).to(input_dtype)  
        
        
        src_frames,tar_frames = top_src_frames,top_tar_frames # Rename to keep consistency...
        ### End of version #2        
        '''
        top_corr_values, top_corr_indices = self._denoiser._force_consistency(soft_correspondences, src_frames[:, :, 0, :], tar_frames[:, :, 0, :])        

        orig_src_embedding = batch['src_embedding'].gather(1, top_src_indices.unsqueeze(-1).expand(-1, -1, 256))
        orig_tar_embedding = batch['tar_embedding'].gather(1, top_tar_indices.unsqueeze(-1).expand(-1, -1, 256))
        virtual_src_coord, src_offsets = self._create_virtual_coordinates(orig_src_embedding, src_frames, src_mask, metadata=batch['metadata'])
        virtual_tar_coord, tar_offsets =  self._create_virtual_coordinates(orig_tar_embedding, tar_frames, tar_mask, metadata=batch['metadata'])
        B, K, _ = top_corr_indices.shape

        # Batch index helper: [B, K]
        batch_indices = torch.arange(B, device=top_corr_indices.device).unsqueeze(-1).expand(-1, K)

        # Gather coordinates
        gathered_virtual_tar = virtual_tar_coord[batch_indices, top_corr_indices[:, :, 0]]  # [B, K, 3]
        gathered_virtual_src = virtual_src_coord[batch_indices, top_corr_indices[:, :, 1]]  # [B, K, 3]
        
        optimal_transformation: dict[str, torch.Tensor] = compute_transformation_from_corr_and_coord(batch['max_length'], top_corr_values, gathered_virtual_src, gathered_virtual_tar, iter_limit=self._max_iter if not self.training else self._n_iter_train)
        return optimal_transformation, src_offsets, tar_offsets

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = move_batch_to_device(batch, self.device)
        # print((batch['metadata'][0]['Ligand_ID'], batch['metadata'][0]['mov_protein'], batch['metadata'][0]['ref_protein'], batch['metadata'][0]['idx']))
        transformation_dict, src_offsets, tar_offsets = self._compute_soft_bb_algorithm(batch)
        
        loss, loss_dict = self._compute_loss(batch, transformation_dict['pred_R'], transformation_dict['pred_t'])
        loss = loss + group_lasso_regularization(src_offsets) + group_lasso_regularization(tar_offsets)
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
        transformation_dict, src_offsets, tar_offsets = self._compute_soft_bb_algorithm(batch)
        loss, loss_dict = self._compute_loss(batch, transformation_dict['pred_R'], transformation_dict['pred_t'])
        loss = loss + group_lasso_regularization(src_offsets) + group_lasso_regularization(tar_offsets)
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
