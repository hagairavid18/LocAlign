from typing import Any
import torch

from models.soft_bb_base import SoftBBBase
from models.utils import move_batch_to_device, build_object, compute_transformation_from_corr_and_coord, mask_and_normalize_matrix


class VirtualSoftBB(SoftBBBase):
    def __init__(self, loss: dict[str, Any], optimizer: dict[str, Any], input_layer, virtual_layer: dict, scalar_layer: dict | None = None, max_iter: int = 5, compute_pocket_importance: bool = False, plot_alignments: bool = False):
       
        super().__init__(loss=loss, optimizer=optimizer, max_iter=max_iter)
        self._input_tar_block = build_object(input_layer, 'models.layers')
        self._input_src_block = build_object(input_layer, 'models.layers')
        self._tar_linear = build_object(scalar_layer, 'models.layers')  # Projects tar_embedding to a scalar
        self._src_linear = build_object(scalar_layer, 'models.layers')
        self._virtual_point_block = build_object(virtual_layer, 'models.layers')
        self._compute_pocket_importance = compute_pocket_importance
    
    def _get_distance_matrix(self, src_embedding: torch.Tensor, tar_embedding: torch.Tensor, src_mask: torch.Tensor, tar_mask: torch.Tensor) -> dict[str, torch.Tensor]:        

        distance_matrix = torch.sqrt(torch.sum((src_embedding.unsqueeze(1) - tar_embedding.unsqueeze(2)) ** 2, dim=-1))
        
        tar_scalar = self._tar_linear(tar_embedding, mask=src_mask).squeeze(-1) 
        src_scalar = self._src_linear(src_embedding, mask=tar_mask).squeeze(-1)
            
        tar_matrix = tar_scalar.unsqueeze(-1).expand_as(distance_matrix)  # Shape [B, N_tar, N_src]
        src_matrix = src_scalar.unsqueeze(1).expand_as(distance_matrix)  # Shape [B, N_tar, N_src]

        distance_matrix = distance_matrix + tar_matrix + src_matrix
            
        return distance_matrix
    
    def _transform_offsets_with_frames(self, offsets: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        """
        Transforms the offsets with the frames of each residue.
        Each predicted offset is transformed with the corresponding frame of the residue, by applying rotation and translation.
        

        Args:
            offsets (torch.Tensor): _description_
            frames (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        transformations = torch.zeros(frames.shape[0], frames.shape[1], 4, 4, device=frames.device)
        transformations[:, :, :3, :3] = frames[:,:, 1:4, :].transpose(2,3)
        transformations[:, :, :3, 3] = frames[:, :, 0, :] # TODO: wait for bugfix

        transformations[:, :, 3, 3] = 1
        offsets = torch.cat([offsets, torch.ones_like(offsets[:, :, :1])], dim=-1)
        return torch.matmul(transformations, offsets.unsqueeze(-1)).squeeze(-1)[..., :3]
    
    def _create_virtual_coordinates(self, embedding: torch.Tensor, frames: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            embedding (torch.Tensor): embedding.
            frames (torch.Tensor): Frames of the residues.
            mask (torch.Tensor): Mask of the residues.

        Returns:
            torch.Tensor: virtual points per input, transformed with the frames.
        """        
        offsets = self._virtual_point_block(embedding, mask)
        transformed_offsets = self._transform_offsets_with_frames(offsets, frames)
        transformed_offsets  = transformed_offsets * mask.unsqueeze(-1)
        return transformed_offsets
    
    def _compute_soft_bb_algorithm(self, batch: dict[torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        The main algorithm of the model. Computes the soft correspondence matrix and the optimal transformation between two sets of embeddings.

        Args:
            batch (dict[torch.Tensor]): Dictionary containing the input tensors.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]: A tuple containing the transformation dictionary and the embeddings dictionary.
        """
        tar_embedding, src_embedding  = self._input_tar_block(batch['tar_embedding'], mask=batch['tar_mask']), self._input_src_block(batch['src_embedding'], mask =batch['src_mask'])
        distance_matrix: torch.Tensor = self._get_distance_matrix(src_embedding=src_embedding, tar_embedding=tar_embedding, src_mask=batch['src_mask'], tar_mask=batch['tar_mask'])
        soft_correspondences, mask_2d = mask_and_normalize_matrix(distance_matrix, batch['src_mask'], batch['tar_mask'], src_embedding, tar_embedding)
        virtual_src_coord, virtual_tar_coord = self._create_virtual_coordinates(src_embedding, batch['src_frames'], batch['src_mask']), self._create_virtual_coordinates(tar_embedding, batch['tar_frames'], batch['tar_mask'])
        optimal_transformation: dict[str, torch.Tensor] = compute_transformation_from_corr_and_coord(batch['max_length'], soft_correspondences, virtual_src_coord, virtual_tar_coord, batch['src_mask'], mask_2d, iter_limit=self._max_iter if not self.training else 2)   
        return optimal_transformation

    def training_step(self, batch: dict[torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = move_batch_to_device(batch, self.device)
        print((batch['metadata'][0]['Ligand_ID'], batch['metadata'][0]['mov_protein'], batch['metadata'][0]['ref_protein'], batch['metadata'][0]['idx']))
        transformation_dict = self._compute_soft_bb_algorithm(batch)
        
        loss, loss_dict = self._compute_loss(batch, transformation_dict['pred_R'], transformation_dict['pred_t'])
        
        outputs = {'loss': loss , 'loss_dict': loss_dict, 'transformation_dict' :transformation_dict}    
        print(outputs['loss_dict'])  
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
        transformation_dict = self._compute_soft_bb_algorithm(batch)

        loss, loss_dict = self._compute_loss(batch, transformation_dict['pred_R'], transformation_dict['pred_t'])
        
        outputs = {'loss': loss , 'loss_dict': loss_dict, 'transformation_dict': transformation_dict}
        self._metrics.update(batch, outputs)
        return outputs
