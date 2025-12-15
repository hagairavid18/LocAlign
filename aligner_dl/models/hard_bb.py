
from datetime import datetime
import os
import torch
from torch.utils.data import DataLoader
import lightning as L
from models.utils.kabsch import kabsch_torch
from models.utils.collate import custom_collate_fn


class HardBB(L.LightningModule):
    def __init__(self):
        super().__init__()
        self._validation_outputs = {}
       
    def validation_step(self, batch: dict[torch.Tensor], batch_idx: int):
        tar_embedding, src_embedding  = batch['tar_embedding'], batch['src_embedding']
        bsize = tar_embedding.shape[0]
        tar_expanded = tar_embedding.unsqueeze(2)  # Shape: (BATCH, A, 1, C)
        src_expanded = src_embedding.unsqueeze(1)  # Shape: (BATCH, 1, B, C)

        # Calculate squared differences
        l2_distances = torch.sqrt(torch.sum((tar_expanded - src_expanded) ** 2, dim=-1))  # Shape: (BATCH, A, B)
        min_indices_tar_to_src = torch.argmin(l2_distances, axis=-1)
        min_indices_src_to_tar = torch.argmin(l2_distances, axis=-2)
        # Find reciprocal pairs
        reciprocal_pairs = []
        for batch_id in range(bsize):
            for a_index in range(tar_embedding.shape[1]):
                b_index = min_indices_tar_to_src[batch_id, a_index].item()
                if min_indices_src_to_tar[batch_id, b_index] == a_index:
                    reciprocal_pairs.append((batch_id, a_index, b_index, l2_distances[batch_id, a_index, b_index]))
        curr_num_pairs = len(reciprocal_pairs) + 1
        
        for batch_id in range(bsize):
            while curr_num_pairs > len(reciprocal_pairs):
                curr_num_pairs = len(reciprocal_pairs)
                if curr_num_pairs <= 10:
                    break
                corresponded_tar_coord = batch['tar_coordinates'][batch_id, [pair[1] for pair in reciprocal_pairs]]
                corresponded_src_coord = batch['src_coordinates'][batch_id, [pair[2] for pair in reciprocal_pairs]]
                weights = [pair[3] for pair in reciprocal_pairs]
                R, t, rmsd, rmsd_per_corr = kabsch_torch(corresponded_src_coord, corresponded_tar_coord)
                print(rmsd)
                indices = torch.where(rmsd_per_corr <= 12)[0].tolist()
                reciprocal_pairs = [reciprocal_pairs[i] for i in indices]
            
            # print(torch.abs(R - torch.Tensor(batch['metadata'][batch_id]['rotations'][0][0])).mean())
            # if len(ransac_R) > 0:
            #     print(torch.abs(torch.Tensor(ransac_R[0]) - torch.Tensor(batch['metadata'][batch_id]['rotations'][0][0])).mean())
            # print(torch.abs(torch.Tensor(batch['metadata'][batch_id]['TMAligner_rotations']) - torch.Tensor(batch['metadata'][batch_id]['rotations'][0][0])).mean())
            # if batch['metadata'][batch_id]['DaliAligner_rotations'].any():
            #     print(torch.abs(torch.Tensor(batch['metadata'][batch_id]['DaliAligner_rotations']) - torch.Tensor(batch['metadata'][batch_id]['rotations'][0][0])).mean())
            # print()
        self._validation_outputs[batch['metadata'][0]['index']] = {
        'svd_R': R,
        'svd_t': t,
        }
     
    
    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
