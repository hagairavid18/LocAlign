from typing import Any
import torch

from models.layers.linear import FeatureBlockGPT
from models.soft_bb_base import SoftBBBase
from models.utils.collate import  move_batch_to_device
from models.utils.misc import build_object
from models.utils.bbs import  compute_transformation_from_corr_and_coord, expand_embeddings_to_atoms, mask_and_normalize_matrix
from models.utils.tensor_operations import create_2d_mask


class SoftBB(SoftBBBase):
    def __init__(self, loss: dict[str, Any], optimizer: dict[str, Any], input_layer , scalar_layer: dict | None = None, use_scalar_layer: bool = False, max_iter: int = 5, compute_pocket_importance: bool = False, plot_alignments: bool = False, use_atom_level=False, use_atom_distances=False):
        """
        SoftBB model class. Generates a soft correspondence matrix between two sets of embeddings and computes the optimal transformation between them.
        Implements a version where the coordinates are the CA atoms of the proteins.

        Args:
            input_layer (_type_): Configuration for the input layer.
            use_scalar_layer (bool, optional): . Defaults to False.
            scalar_layer (dict | None, optional): configuration for the scalar_layer that predict confidence per residue. Defaults to None.
            compute_pocket_importance (bool, optional): _description_. Defaults to False.
            use_atom_level (bool, optional): _description_. Defaults to False.
            use_atom_distances (bool, optional): _description_. Defaults to False.
        """        
        super().__init__(loss=loss, optimizer=optimizer, max_iter=max_iter, use_atom_level=use_atom_level)
        self._use_scalar_layer = use_scalar_layer
        self._input_tar_block = build_object(input_layer, 'models.layers')
        self._input_src_block = build_object(input_layer, 'models.layers')
        if use_scalar_layer:
            self._tar_linear = build_object(scalar_layer, 'models.layers')  # Projects tar_embedding to a scalar
            self._src_linear = build_object(scalar_layer, 'models.layers')
        self._pocket_loss = build_object(loss['pocket'], 'losses')
        self._compute_pocket_importance = compute_pocket_importance
        if self._use_atom_level:
            self._atom_embedding_block = FeatureBlockGPT(128, 128, 128, n_blocks=1, skip_connection=True)
        self._use_atom_distances = use_atom_distances
    
    def create_correspondences_matrix(self, batch, tar_embedding: torch.Tensor, src_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        ret_dict = {}
        tar_embedding, src_embedding  = self._input_tar_block(tar_embedding, mask =batch['tar_mask']), self._input_src_block(src_embedding, mask =batch['src_mask'])
        # tar_embedding, src_embedding  = self._input_tar_block(tar_embedding), self._input_src_block(src_embedding)
        print(f"tar_embedding mean after layer: {tar_embedding.mean()}, tar_embedding std: {tar_embedding.std()}")
        if self._use_atom_level:
            # Expand embeddings to match atom-level resolution
            src_embedding = expand_embeddings_to_atoms(
                batch["src_all_coordinates"],
                batch["src_residue_indices"],
                src_embedding,
                batch["src_all_mask"],
                batch["src_mask"]
            )
            tar_embedding = expand_embeddings_to_atoms(
                batch["tar_all_coordinates"],
                batch["tar_residue_indices"],  # Use the same logic to align target indices, if applicable
                tar_embedding,
                batch['tar_all_mask'],
                batch['tar_mask']
            )
            src_embedding = self._atom_embedding_block(src_embedding)
            tar_embedding = self._atom_embedding_block(tar_embedding)
        l2_embedding = torch.sqrt(torch.sum((src_embedding.unsqueeze(1) - tar_embedding.unsqueeze(2)) ** 2, dim=-1))
        
        if self._use_scalar_layer:
            tar_scalar = self._tar_linear(tar_embedding, mask=batch['tar_mask']).squeeze(-1) 
            src_scalar = self._src_linear(src_embedding, mask=batch['src_mask']).squeeze(-1)
                
            tar_matrix = tar_scalar.unsqueeze(-1).expand_as(l2_embedding)  # Shape [B, N_tar, N_src]
            src_matrix = src_scalar.unsqueeze(1).expand_as(l2_embedding)  # Shape [B, N_tar, N_src]

            ret_dict.update({"l2_embedding" :l2_embedding + tar_matrix + src_matrix, "tar_embedding": tar_embedding, "src_embedding": src_embedding})
        else:
            ret_dict.update({"l2_embedding" :l2_embedding, "tar_embedding": tar_embedding, "src_embedding": src_embedding})
        return ret_dict
    
    
    def _compute_soft_bb_algorithm(self, batch: dict[torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        combined_mask = create_2d_mask(batch['src_mask'], batch['tar_mask'])
        embeddings_dict, l2_embedding = self.compute_correspondences(batch, combined_mask)
        print(f"given to mask and normalize 0,0: {l2_embedding[0,0,0]}")
        soft_correspondences = mask_and_normalize_matrix(l2_embedding, batch['src_mask'], batch['tar_mask'], embeddings_dict['src_embedding'], embeddings_dict['tar_embedding']) * combined_mask
        print(f"soft_correspondences mean: {soft_correspondences.mean()}, soft_correspondences std: {soft_correspondences.std()}")
        src_coordinates = batch['src_all_coordinates'][...,:3] if self._use_atom_level else batch['src_coordinates']
        tar_coordinates = batch['src_all_coordinates'][...,:3] if self._use_atom_level else batch['tar_coordinates']
                
        transformation_dict = compute_transformation_from_corr_and_coord(batch['max_length'], soft_correspondences, src_coordinates, tar_coordinates, batch['src_mask'], combined_mask, iter_limit=self._max_iter if not self.training else 2)   
        return transformation_dict, embeddings_dict

    def training_step(self, batch: dict[torch.Tensor]):
        pair_info = (batch['metadata'][0]['Ligand_ID'], batch['metadata'][0]['mov_protein'], batch['metadata'][0]['ref_protein'], batch['metadata'][0]['bbr'][0][0])
        print(pair_info)
        print(f"src_embedding mean: {batch['src_embedding'].mean()}, src_embedding std: {batch['src_embedding'].std()}")
        batch = move_batch_to_device(batch, self.device)
        transformation_dict, embeddings_dict = self._compute_soft_bb_algorithm(batch)
        
        loss, loss_dict = self._compute_loss(batch, transformation_dict['pred_R'], transformation_dict['pred_t'])
        
        outputs = {'loss': loss , 'loss_dict': loss_dict, 'transformation_dict' :transformation_dict}      
        return outputs

    def validation_step(self, batch):
        batch = move_batch_to_device(batch, self.device)
        transformation_dict, embeddings_dict = self._compute_soft_bb_algorithm(batch)

        loss, loss_dict = self._compute_loss(batch, transformation_dict['pred_R'], transformation_dict['pred_t'])
        
        outputs = {'loss': loss , 'loss_dict': loss_dict, 'transformation_dict': transformation_dict, "embeddings_dict": embeddings_dict}
        self._metrics.update(batch, outputs)
        return outputs
