
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
        super().__init__(loss=None, optimizer=None)
        self._metrics = PocketRMSD()
        self._name = model_name

    def validation_step(self, batch, batch_idx):
        batch = move_batch_to_device(batch, self.device)
        batch_size = len(batch['metadata'])
        device = self.device
        assert batch_size == 1
        R = torch.Tensor(batch['metadata'][0][f'{self._name}_rotations']).to(device)
        t = torch.Tensor(batch['metadata'][0][f'{self._name}_translations']).to(device)

        if R.shape[0] == 0 or t.shape[0] == 0: # Handle empty tensors, create B x 3 identity rotation and zero translation
            R = torch.eye(3, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
            t = torch.zeros(batch_size, 3, device=self.device)        
        outputs = { 'transformation_dict': {'pred_R': R, 'pred_t': t, 'all_R': [R], 'all_t': [t]} }
        self._metrics.update(batch, outputs)
        return outputs
    
    def training_step(self):
        pass

    def configure_optimizers(self):
        pass


        