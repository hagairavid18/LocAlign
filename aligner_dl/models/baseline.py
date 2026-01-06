
from datetime import datetime
import os
import torch
import lightning as L

from models.soft_bb_base import SoftBBBase
from models.utils.collate import custom_collate_fn, move_batch_to_device
from metrics import PocketRMSD
from models.utils.misc import build_object, flatten_dict


class Baseline(SoftBBBase):
    def __init__(self, model_name: str) -> None:
        super().__init__(loss=None, optimizer=None, metric=None)
        self._metrics = PocketRMSD()
        self._name = model_name

    def validation_step(self, batch, batch_idx):
        batch = move_batch_to_device(batch, self.device)
        batch_size = len(batch['metadata'])
        device = self.device
        assert batch_size == 1
        
        metadata = batch['metadata'][0]
        
        # Extract precomputed values from metadata
        ligand_rmsd_val = float(metadata[f'{self._name}_ligand_rmsd'])
        corr_rmsd_val = float(metadata[f'{self._name}_corr_rmsd'])
        
        # Create pseudo loss_dict with precomputed values from metadata
        pseudo_loss_dict = {
            'per_sample': {
                'ligand_rmsd': torch.tensor([ligand_rmsd_val], device=device),
                'corr_rmsd': torch.tensor([corr_rmsd_val], device=device),
                'embedding': torch.tensor([0.0], device=device),
                'gap': torch.tensor([0.0], device=device),
                'radius': torch.tensor([0.0], device=device),
            }
        }
        
        outputs = { 
            'transformation_dict': {'pred_R': None, 'pred_t': None},
            'loss_dict': pseudo_loss_dict
        }
        self._metrics.update(batch, outputs)
        return outputs
    
    def training_step(self):
        pass

    def configure_optimizers(self):
        pass


        