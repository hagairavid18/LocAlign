from typing import Any
import torch

from models.soft_bb_base import SoftBBBase
from models.utils import move_batch_to_device, build_object, compute_transformation_from_corr_and_coord, mask_and_normalize_matrix
from models.utils.math import group_lasso_regularization
from models.utils.plots import plot_correspondences, plot_offsets
torch.set_float32_matmul_precision('medium')


class VirtualSoftBB(SoftBBBase):
    def __init__(self, loss: dict[str, Any], optimizer: dict[str, Any], input_layer: dict[str, Any], virtual_layer: dict, scalar_layer: dict, denoiser: dict, max_iter: int = 5, n_iter_train: int = 2, plot_dir: str | None = None) -> None:
       
        super().__init__(loss=loss, optimizer=optimizer, max_iter=max_iter, n_iter_train=n_iter_train, plot_dir=plot_dir)
        self._input_block = build_object(input_layer, 'models.layers')
        self._linear = build_object(scalar_layer, 'models.layers')  # Projects tar_embedding to a scalar
        self._virtual_point_block = build_object(virtual_layer, 'models.layers')
        self._denoiser = build_object(denoiser, 'models')
        # self._automatic_optimization = False
    
    def _get_distance_matrix(self, src_embedding: torch.Tensor, tar_embedding: torch.Tensor, src_mask: torch.Tensor, tar_mask: torch.Tensor) -> torch.Tensor: 
        dtype = src_embedding.dtype
        dim = src_embedding.shape[-1]
        scale = torch.sqrt(torch.tensor(dim, device=src_embedding.device, dtype=src_embedding.dtype))
        diff = src_embedding.unsqueeze(1) - tar_embedding.unsqueeze(2)
        distance_matrix = torch.sum(torch.mul(diff, diff), dim=-1) / scale

        tar_scalar = self._linear(tar_embedding, mask=tar_mask).squeeze(-1).to(dtype)
        src_scalar = self._linear(src_embedding, mask=src_mask).squeeze(-1).to(dtype)
            
        tar_matrix = tar_scalar.unsqueeze(-1).expand_as(distance_matrix)  # Shape [B, N_tar, N_src]
        src_matrix = src_scalar.unsqueeze(1).expand_as(distance_matrix)  # Shape [B, N_tar, N_src]

        distance_matrix = distance_matrix + tar_matrix + src_matrix
            
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
        tar_embedding, src_embedding  = self._input_block(batch['tar_embedding'], mask=batch['tar_mask']).to(input_dtype), self._input_block(batch['src_embedding'], mask =batch['src_mask']).to(input_dtype)
        # tar_embedding, src_embedding  = batch['tar_embedding'], batch['src_embedding']
        distance_matrix: torch.Tensor = self._get_distance_matrix(src_embedding=src_embedding, tar_embedding=tar_embedding, src_mask=batch['src_mask'], tar_mask=batch['tar_mask'])
        distance_matrix = distance_matrix.to(input_dtype)
        soft_correspondences, mask_2d = mask_and_normalize_matrix(distance_matrix, batch['src_mask'], batch['tar_mask'], src_embedding, tar_embedding)
        soft_correspondences = soft_correspondences.to(input_dtype)
        updated_correspondences, top_k_indices = self._denoiser._force_consistency(soft_correspondences, batch['src_frames'][:, :, 0, :], batch['tar_frames'][:, :, 0, :])        
        # virtual_src_coord, virtual_tar_coord = batch['src_frames'][:, :, 0, :], batch['tar_frames'][:, :, 0, :]
        virtual_src_coord, src_offsets = self._create_virtual_coordinates(src_embedding, batch['src_frames'], batch['src_mask'], metadata=batch['metadata'])
        virtual_tar_coord, tar_offsets =  self._create_virtual_coordinates(tar_embedding, batch['tar_frames'], batch['tar_mask'], metadata=batch['metadata'])
        B, K, _ = top_k_indices.shape

        # Split indices
        tar_idx = top_k_indices[:, :, 0]  # [B, K]
        src_idx = top_k_indices[:, :, 1]  # [B, K]

        # Batch index helper: [B, K]
        batch_indices = torch.arange(B, device=top_k_indices.device).unsqueeze(-1).expand(-1, K)

        # Gather coordinates
        gathered_virtual_tar = virtual_tar_coord[batch_indices, tar_idx]  # [B, K, 3]
        gathered_virtual_src = virtual_src_coord[batch_indices, src_idx]  # [B, K, 3]

        # virtual_src_coord, virtual_tar_coord = batch['src_frames'][:, :, 0, :], batch['tar_frames'][:, :, 0, :]
        optimal_transformation: dict[str, torch.Tensor] = compute_transformation_from_corr_and_coord(batch['max_length'], updated_correspondences, gathered_virtual_src, gathered_virtual_tar, iter_limit=self._max_iter if not self.training else self._n_iter_train)
        return optimal_transformation, mask_2d, src_offsets, tar_offsets

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = move_batch_to_device(batch, self.device)
        # print((batch['metadata'][0]['Ligand_ID'], batch['metadata'][0]['mov_protein'], batch['metadata'][0]['ref_protein'], batch['metadata'][0]['idx']))
        transformation_dict, _, src_offsets, tar_offsets = self._compute_soft_bb_algorithm(batch)
        
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
        transformation_dict, mask, src_offsets, tar_offsets = self._compute_soft_bb_algorithm(batch)
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
