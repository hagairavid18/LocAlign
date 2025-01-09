from typing import Any
import torch

from models.soft_bb_base import SoftBBBase
from models.utils import move_batch_to_device, build_object, compute_transformation_from_corr_and_coord, mask_and_normalize_matrix

class SoftBB(SoftBBBase):
    def __init__(self, loss: dict[str, Any], optimizer: dict[str, Any], input_layer, scalar_layer: dict | None = None, max_iter: int = 5, compute_pocket_importance: bool = False, plot_alignments: bool = False):
        """
        SoftBB model class. Generates a soft correspondence matrix between two sets of embeddings and computes the optimal transformation between them.
        Implements a version where the coordinates are the CA atoms of the proteins.

        Args:
            input_layer (_type_): Configuration for the input layer.
            scalar_layer (dict | None, optional): configuration for the scalar_layer that predict confidence per residue. Defaults to None.
            compute_pocket_importance (bool, optional): _description_. Defaults to False.
        """        
        super().__init__(loss=loss, optimizer=optimizer, max_iter=max_iter)
        self._input_tar_block = build_object(input_layer, 'models.layers')
        self._input_src_block = build_object(input_layer, 'models.layers')
        self._tar_linear = build_object(scalar_layer, 'models.layers')  # Projects tar_embedding to a scalar
        self._src_linear = build_object(scalar_layer, 'models.layers')
        self._pocket_loss = build_object(loss['pocket'], 'losses')
        self._compute_pocket_importance = compute_pocket_importance
    
    def _get_distance_matrix(self, batch, src_embedding: torch.Tensor, tar_embedding: torch.Tensor) -> dict[str, torch.Tensor]:        

        distance_matrix = torch.sqrt(torch.sum((src_embedding.unsqueeze(1) - tar_embedding.unsqueeze(2)) ** 2, dim=-1))
        
        tar_scalar = self._tar_linear(tar_embedding, mask=batch['tar_mask']).squeeze(-1) 
        src_scalar = self._src_linear(src_embedding, mask=batch['src_mask']).squeeze(-1)
            
        tar_matrix = tar_scalar.unsqueeze(-1).expand_as(distance_matrix)  # Shape [B, N_tar, N_src]
        src_matrix = src_scalar.unsqueeze(1).expand_as(distance_matrix)  # Shape [B, N_tar, N_src]

        distance_matrix = distance_matrix + tar_matrix + src_matrix
            
        return distance_matrix
    
    def _compute_soft_bb_algorithm(self, batch: dict[torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        The main algorithm of the model. Computes the soft correspondence matrix and the optimal transformation between two sets of embeddings.

        Args:
            batch (dict[torch.Tensor]): Dictionary containing the input tensors.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]: A tuple containing the transformation dictionary and the embeddings dictionary.
        """
        src_coordinates, tar_coordinates = batch['src_frames'][:, :, 0, :], batch['tar_frames'][:, :, 0, :]     
        tar_embedding, src_embedding  = self._input_tar_block(batch['tar_embedding'], mask=batch['tar_mask']), self._input_src_block(batch['src_embedding'], mask =batch['src_mask'])
        distance_matrix: torch.Tensor = self._get_distance_matrix(batch, src_embedding=src_embedding, tar_embedding=tar_embedding)
        soft_correspondences, mask_2d = mask_and_normalize_matrix(distance_matrix, batch['src_mask'], batch['tar_mask'], src_embedding, tar_embedding)
        optimal_transformation: dict[str, torch.Tensor] = compute_transformation_from_corr_and_coord(batch['max_length'], soft_correspondences, src_coordinates, tar_coordinates, batch['src_mask'], mask_2d, iter_limit=self._max_iter if not self.training else 2)   
        return optimal_transformation

    def training_step(self, batch: dict[torch.Tensor]):
        batch = move_batch_to_device(batch, self.device)
        transformation_dict = self._compute_soft_bb_algorithm(batch)
        
        loss, loss_dict = self._compute_loss(batch, transformation_dict['pred_R'], transformation_dict['pred_t'])
        
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
        batch = move_batch_to_device(batch, self.device)
        transformation_dict = self._compute_soft_bb_algorithm(batch)

        loss, loss_dict = self._compute_loss(batch, transformation_dict['pred_R'], transformation_dict['pred_t'])
        
        outputs = {'loss': loss , 'loss_dict': loss_dict, 'transformation_dict': transformation_dict}
        self._metrics.update(batch, outputs)
        return outputs
