
from datetime import datetime
import os
import torch
import lightning as L

from models.soft_bb_base import SoftBBBase
from models.utils.collate import custom_collate_fn, move_batch_to_device
from metrics import PocketRMSD
from models.utils.misc import build_object, flatten_dict


class TMaligner(SoftBBBase):
    def __init__(self):
        super().__init__(loss=None, optimizer=None)
        self._metrics = PocketRMSD()

    def validation_step(self, batch, batch_idx):
        batch = move_batch_to_device(batch, self.device)
        assert len(batch['metadata']) == 1
        R = torch.Tensor(batch['metadata'][0]['TMaligner_rotations'])
        t = torch.Tensor(batch['metadata'][0]['TMaligner_translations'])        
        outputs = { 'transformation_dict': {'pred_R': R, 'pred_t': t}}
        self._metrics.update(batch, outputs)
        return outputs
    
    def training_step(self):
        pass

    def configure_optimizers(self):
        pass


if __name__ == "__main__":
    import logging
    import yaml
    from torch.utils.data import DataLoader
    from pytorch_lightning.loggers import CometLogger

    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", "learnable_softbbs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, start_time + ".log"), level=logging.INFO, format='%(message)s')

    logger = logging.getLogger(__name__)
    

    with open('configs/tm_align.yaml') as f:
        config = yaml.safe_load(f)

    valid_dataset = build_object(config['dataset']['validation'], 'datasets')
    val_loader  = DataLoader(valid_dataset, batch_size=config['dataloader']['valid_batch_size'], collate_fn=custom_collate_fn, num_workers=20)

    model = build_object(config['model'], 'models')

    comet_logger = CometLogger(
        api_key="9ydBzigeK75Z6RhAiX63xGdsg",
        workspace="hagairavid18",
        project_name="pocket_aligner",
        experiment_name=config['trainer']['exp_name']
    )

    trainer = L.Trainer(logger=comet_logger, 
                        accelerator= 'gpu' if torch.cuda.is_available() else 'cpu', profiler="simple")
        
    trainer.logger.log_hyperparams(flatten_dict(config))

    trainer.validate(model, val_loader)
        