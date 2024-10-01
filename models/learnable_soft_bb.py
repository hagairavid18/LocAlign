
from datetime import datetime
import os
from typing import Any
import torch
from torch.utils.data import DataLoader
import lightning as L
from pytorch_lightning.loggers import CometLogger
import torch.optim as optim

from models.soft_bb import SoftBB
from utils.constants import LIGAND_DIR
from utils.kabsch import weighted_kabsch_torch
from models.utils.collate import custom_collate_fn, move_batch_to_device
from utils.deepbbs_utils import *
from utils.transformation import rotation_matrix_to_euler_angles
from models.layers import FeatureBlock
from metrics import PocketRMSD
from utils.misc import build_object

from utils.plots import plot_transformed_point_clouds

torch.set_float32_matmul_precision('medium')


class LearnableSoftBB(L.LightningModule):
    def __init__(self, loss_config: dict[str, Any], lr: float = 1e-3, max_iter: int = 5, n_linear_blocks: int = 3):
        super().__init__()
        self._validation_outputs = {}
        self._inout_tar_block = FeatureBlock(128 ,128, 128, n_linear_blocks)
        self._inout_src_block = FeatureBlock(128 ,128, 128, n_linear_blocks)
        self._pocket_loss = build_object(loss_config['pocket'], 'losses')
        self._transformation_loss = build_object(loss_config['transformation'], 'losses')
        self._alpha_loss = 0.5
        self._metrics = PocketRMSD()
        self.validation_step_outputs = []
        self.eps = 0.00001
        self._min_diff = 0.05
        self._max_iter = max_iter
        self._lr = lr
    
    def get_d0(self, max_length) -> torch.Tensor:
        return 1.24*(torch.tensor(max_length, device=self.device) - 15)**(1/3) -1.8
    
    def _compose_transformations(self, batch_size, rotations: list[torch.Tensor], translations: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        R_total = torch.eye(3, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1) 
        t_total = torch.zeros(batch_size, 3, device=self.device)

        for R, t in zip(rotations, translations):
            R_total = R_total @ R
            t_total = torch.bmm(t_total.unsqueeze(1), R).squeeze(1) + t

        return R_total, t_total
    
    def create_2d_mask(self, batch) -> torch.Tensor:
        batch_size = batch['tar_embedding'].shape[0]
        mask_dim1_expanded = batch['tar_mask'].unsqueeze(2).expand(batch_size, -1, 1000) 
        mask_dim2_expanded = batch['src_mask'].unsqueeze(1).expand(batch_size, 1000, -1)  
        combined_mask = mask_dim1_expanded * mask_dim2_expanded
        return combined_mask

    def _mask_and_normalize_matrix(self, distance_matrix: torch.Tensor, src_mask: torch.Tensor, tar_mask: torch.Tensor, src_embedding: torch.Tensor, tar_embedding) -> list[tuple]:
        batch_size = distance_matrix.shape[0]
        t = torch.tensor([guess_best_alpha_torch(src_embedding[i,:][src_mask[i]], dim_num=tar_embedding.shape[-1], transpose=False) for i in range(batch_size)], device=self.device)
        R = torch.stack([softargmin_rows_torch(distance_matrix[i], t[i]) for i in range(batch_size)], dim=0)
        t = torch.tensor([guess_best_alpha_torch(tar_embedding[i,:][tar_mask[i]], dim_num=tar_embedding.shape[-1], transpose=False) for i in range(batch_size)] , device=self.device)
        C = torch.stack([softargmin_rows_torch(torch.transpose(distance_matrix, dim0=1, dim1=2)[i], t[i]) for i in range(batch_size)], dim=0)
        C = torch.transpose(C, dim0=1, dim1=2)
        B = torch.mul(R, C)
        return B
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        batch_size = batch['tar_embedding'].shape[0]
        for loss_name, value in outputs['loss_dict'].items():
            self.log(f'train_{loss_name}_loss', value, batch_size=batch_size, prog_bar=False, on_step=True, on_epoch=True)
        self.log(f'train_loss', outputs['loss'], batch_size=batch_size, prog_bar=False, on_step=True, on_epoch=True)
        # transformations_to_plot = {"GT" : (batch['gt_R'], batch['gt_t'][:,:3]),
        #                             "TMalign": (torch.Tensor(batch['metadata'][0]['TMaligner_rotations']), torch.Tensor(batch['metadata'][0]['TMaligner_translations'])),
        #                             "iterative_SoftBBS" :(outputs['pred_R'].clone().detach(), outputs['pred_t'].clone().detach())}
        # if batch_idx % 50 == 0:
        #     logger.experiment.add_image("Transformed Point Clouds", plot_transformed_point_clouds(batch, transformations_to_plot,  compose = False), self.global_step)
    
    def on_validation_epoch_end(self):
        metrics = self._metrics.compute()
        self.log('valid_pocket_rmsd', metrics['pocket_rmsd'], on_epoch=True)
        self.log('valid_ligand_rmsd', metrics['ligand_rmsd'], on_epoch=True)
        self.log('valid_pocket_rmsd_iter0', metrics['pocket_rmsd_iter0'], on_epoch=True)
        self.log('valid_ligand_rmsd_iter0', metrics['ligand_rmsd_iter0'], on_epoch=True)
        self._metrics.reset()
    
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        batch_size = batch['tar_embedding'].shape[0]
        for loss_name, value in outputs['loss_dict'].items():
            self.log(f'valid_{loss_name}_loss', value, batch_size=batch_size, prog_bar=False, on_epoch=True)
        self.log(f'valid_loss', outputs['loss'], batch_size=batch_size, prog_bar=False, on_epoch=True)
        # transformations_to_plot = {"GT" : (batch['gt_R'], batch['gt_t'][:,:3]),
        #                             "TMalign": (torch.Tensor(batch['metadata'][0]['TMaligner_rotations']), torch.Tensor(batch['metadata'][0]['TMaligner_translations'])),
        #                             "iterative_SoftBBS" :(outputs['pred_R'].clone().detach(), outputs['pred_t'].clone().detach())}
        # if batch_idx % 50 == 0:
        #     logger.experiment.add_image("Transformed Point Clouds", plot_transformed_point_clouds(batch, transformations_to_plot,  compose = False), self.global_step)
    
    def training_step(self, batch: dict[torch.Tensor], batch_idx: int):
        batch = move_batch_to_device(batch, self.device)
        batch_size = batch['tar_embedding'].shape[0]
        tar_embedding, src_embedding  = self._inout_tar_block(batch['tar_embedding']), self._inout_src_block(batch['src_embedding'])
        combined_mask = self.create_2d_mask(batch)

        l2_embedding = torch.sqrt(torch.sum((src_embedding.unsqueeze(1) - tar_embedding.unsqueeze(2)) ** 2, dim=-1))
        l2_embedding = l2_embedding * combined_mask
        l2_embedding = l2_embedding.masked_fill(~combined_mask, float('inf'))
        gamma_0 = self._mask_and_normalize_matrix(l2_embedding, batch['src_mask'], batch['tar_mask'], src_embedding, tar_embedding) * combined_mask

        gamma = gamma_0.clone()
        src_coordinates = batch['src_coordinates']
        all_R, all_t = [], []
        iter_num = 0
        while(iter_num < 2):
            R_gamma, t_gamma, _, _ = weighted_kabsch_torch(src_coordinates,  batch['tar_coordinates'], gamma.float(),  batch['src_mask'],  batch['tar_mask'])
            all_R.append(R_gamma)
            all_t.append(t_gamma)
            src_coordinates = (torch.matmul(src_coordinates, R_gamma) + t_gamma.unsqueeze(1)) * batch['src_mask'].unsqueeze(-1).expand_as(batch['src_coordinates'])
            
            src_tgt_euc_dist = cdist_torch(batch['tar_coordinates'], src_coordinates, 3)
            gamma = (gamma_0 / (( 1 + (src_tgt_euc_dist / self.get_d0(batch['max_length'])[:, None, None])**2)**2)) * combined_mask
            iter_num +=1
        
        R_total, t_total = self._compose_transformations(batch_size, all_R, all_t)
        
        loss, loss_dict = self._compute_loss(batch, R_total, t_total)
        
        outputs = {'loss': loss , 'loss_dict': loss_dict, 'pred_R': R_total, 'pred_t': t_total, 'pred_first_R': all_R[0], 'pred_first_t': all_t[0]}        
        return outputs
    
    def _compute_loss(self, batch, R_total, t_total):
        pocket_loss: dict[str, torch.Tensor] = self._pocket_loss(batch, R_total, t_total)
        transformation_loss: dict[str, torch.Tensor] = self._transformation_loss(batch, R_total, t_total)
        combined_loss = self._alpha_loss * pocket_loss['pocket_rmsd'] + (1-self._alpha_loss) * transformation_loss['transformation']
        loss_dict = pocket_loss | transformation_loss
        loss_dict['loss'] = combined_loss
        return combined_loss, loss_dict

    def validation_step(self, batch, batch_idx):
        batch = move_batch_to_device(batch, self.device)
        batch_size = batch['tar_embedding'].shape[0]
        tar_embedding, src_embedding  = self._inout_tar_block(batch['tar_embedding']), self._inout_src_block(batch['src_embedding'])
        combined_mask = self.create_2d_mask(batch)

        l2_embedding = torch.sqrt(torch.sum((src_embedding.unsqueeze(1) - tar_embedding.unsqueeze(2)) ** 2, dim=-1))
        l2_embedding = l2_embedding * combined_mask
        l2_embedding = l2_embedding.masked_fill(~combined_mask, float('inf'))
        gamma_0 = self._mask_and_normalize_matrix(l2_embedding, batch['src_mask'], batch['tar_mask'], src_embedding, tar_embedding) * combined_mask
        gamma = gamma_0.clone()
        src_coordinates = batch['src_coordinates']
        all_R, all_t = [], []
        not_converged = True
        iter_num = 1
        while(not_converged):
            R_gamma, t_gamma, rmsd_gamma, rmsd_per_corr_gamma = weighted_kabsch_torch(src_coordinates,  batch['tar_coordinates'], gamma.float(),  batch['src_mask'],  batch['tar_mask'])
            all_R.append(R_gamma)
            all_t.append(t_gamma)
            euler_angles = rotation_matrix_to_euler_angles(R_gamma)
            diff = torch.mean(euler_angles.abs())
            if diff < self._min_diff or iter_num >= self._max_iter:
                not_converged = False
            src_coordinates = (torch.matmul(src_coordinates, R_gamma) + t_gamma.unsqueeze(1)) * batch['src_mask'].unsqueeze(-1).expand_as(batch['src_coordinates'])
            
            src_tgt_euc_dist = cdist_torch(batch['tar_coordinates'], src_coordinates, 3)
            gamma = (gamma_0 / (( 1 + (src_tgt_euc_dist / self.get_d0(batch['max_length'])[:, None, None])**2)**2))* combined_mask
            iter_num +=1        
           
        R_total, t_total = self._compose_transformations(batch_size, all_R, all_t)
        loss, loss_dict = self._compute_loss(batch, R_total, t_total)
        
        outputs = {'loss': loss , 'loss_dict': loss_dict, 'pred_R': R_total, 'pred_t': t_total, 'pred_first_R': all_R[0], 'pred_first_t': all_t[0]}
        self._metrics.update(batch, outputs)
        return outputs

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self._lr)
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'valid_loss',  # Monitors validation loss or another metric
                'interval': 'epoch',      # Frequency to update the scheduler ('epoch' or 'step')
                'frequency': 1,           # Frequency of calling the scheduler
            }
        }
    
    @property
    def automatic_optimization(self):
        return True
    

if __name__ == "__main__":
    from datasets import ScannetDataset
    import logging
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", "learnable_softbbs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, start_time + ".log"), level=logging.INFO, format='%(message)s')

    logger = logging.getLogger(__name__)
    
    config = {
        'learning_rate': 7.5e-4,
        'train_batch_size': 8,
        'valid_batch_size': 1,
        'optimizer': 'Adam',
        'min_cath': 3,
        'max_iter': 5,
        'n_blocks': 2,
        'loss': {'pocket' : {'name': 'PocketLoss', 'args': {}}, 'transformation': {'name': 'RTLoss', 'args': {}}}
    }
    train_dataset = ScannetDataset('results/baseline_results/2024-07-18_10-26-26_10000_train.csv', LIGAND_DIR, n_samples=20000, min_cath = config['min_cath'])
    train_loader  = DataLoader(train_dataset, batch_size=config['train_batch_size'], collate_fn=custom_collate_fn, num_workers=20)
    valid_dataset = ScannetDataset('results/baseline_results/2024-07-18_10-26-26_10000_validation.csv', LIGAND_DIR, min_cath = config['min_cath'])
    val_loader  = DataLoader(valid_dataset, batch_size=config['valid_batch_size'], collate_fn=custom_collate_fn, num_workers=20)
        # model = SoftBB()

        # comet_logger = CometLogger(
        #     api_key="9ydBzigeK75Z6RhAiX63xGdsg",
        #     workspace="hagairavid18", # Optional
        #     project_name="pocket_aligner", # Optional
        #     experiment_name="non-learnable-softbbs-test" # Optional
        #     # rest_api_key=os.environ["COMET_REST_KEY"], # Optional
        # )


        # trainer = L.Trainer(logger=comet_logger, max_epochs=20, log_every_n_steps=25, accelerator= 'gpu' if torch.cuda.is_available() else 'cpu', profiler="simple")
        # trainer.validate(model,  dataloaders=val_loader)
    model = LearnableSoftBB(loss_config=config['loss'], lr=config['learning_rate'], max_iter=config['max_iter'], n_linear_blocks=config['n_blocks'])

    comet_logger = CometLogger(
        api_key="9ydBzigeK75Z6RhAiX63xGdsg",
        workspace="hagairavid18", # Optional
        project_name="pocket_aligner", # Optional
        experiment_name="softbbs-combined-loss" # Optional
        # rest_api_key=os.environ["COMET_REST_KEY"], # Optional
    )

    trainer = L.Trainer(logger=comet_logger, max_epochs=50, log_every_n_steps=100, accelerator= 'gpu' if torch.cuda.is_available() else 'cpu', profiler="simple", check_val_every_n_epoch=1)
    trainer.logger.log_hyperparams(config)

    trainer.fit(model, train_loader, val_dataloaders=val_loader)
        