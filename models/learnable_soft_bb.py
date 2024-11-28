
from datetime import datetime
import os
from typing import Any
import torch
import lightning as L
import torch.optim as optim

from models.utils.collate import custom_collate_fn, move_batch_to_device
from utils.kabsch import weighted_kabsch_torch
from utils.deepbbs_utils import *
from utils.transformation import rotation_matrix_to_euler_angles
from metrics import PocketRMSD
from models.utils.misc import build_object, flatten_dict
from models.utils.bbs import compute_mean_scalar_pocket_values

from models.utils.plots import  generate_and_log_scatter_plot, log_histograms

torch.set_float32_matmul_precision('medium')


class LearnableSoftBB(L.LightningModule):
    def __init__(self, loss: dict[str, Any], optimizer: dict[str, Any], input_layer , scalar_layer, use_scalar_layer: bool = True, max_iter: int = 5, compute_pocket_importance: bool = False):
        super().__init__()
        self._validation_outputs = {}
        self._use_scalar_layer = use_scalar_layer
        self._input_tar_block = build_object(input_layer, 'layers')
        self._input_src_block = build_object(input_layer, 'layers')
        if use_scalar_layer:
            self._tar_linear = build_object(scalar_layer, 'layers')  # Projects tar_embedding to a scalar
            self._src_linear = build_object(scalar_layer, 'layers')
        self._pocket_loss = build_object(loss['pocket'], 'losses')
        self._transformation_loss = None
        if 'transformation' in loss:
            self._transformation_loss = build_object(loss['transformation'], 'losses')
        self._alpha_loss = 0.5
        self._metrics = PocketRMSD()
        self.validation_step_outputs = []
        self.eps = 0.00001
        self._min_diff = 0.05
        self._max_iter = max_iter
        self._lr = optimizer['args']['learning_rate']
        self._compute_pocket_importance = compute_pocket_importance
    
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
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=True, on_epoch=True, logger=True)
        # transformations_to_plot = {"GT" : (batch['gt_R'], batch['gt_t'][:,:3]),
        #                             "TMalign": (torch.Tensor(batch['metadata'][0]['TMaligner_rotations']), torch.Tensor(batch['metadata'][0]['TMaligner_translations'])),
        #                             "iterative_SoftBBS" :(outputs['pred_R'].clone().detach(), outputs['pred_t'].clone().detach())}
        # if batch_idx % 50 == 0:
        #     logger.experiment.add_image("Transformed Point Clouds", plot_transformed_point_clouds(batch, transformations_to_plot,  compose = False), self.global_step)
    
    def on_validation_epoch_end(self):
        metrics = self._metrics.compute()
        
        metric_types = {
            'pocket_rmsd': 'valid_pocket_rmsd',
            'ligand_rmsd': 'valid_ligand_rmsd',
            'pocket_rmsd_iter0': 'valid_pocket_rmsd_iter0',
            'src_pocket_embeddings_scalar':'src_pocket_embeddings_scalar',
            'tar_pocket_embeddings_scalar':'tar_pocket_embeddings_scalar',
            'src_non_pocket_embeddings_scalar':'src_non_pocket_embeddings_scalar',
            'tar_non_pocket_embeddings_scalar':'tar_non_pocket_embeddings_scalar',
        }
        
        # Log total metrics
        for metric_key, log_name in metric_types.items():
            total_value = sum(metrics[metric_key].values()) / len(metrics[metric_key])
            self.log(log_name, total_value, on_epoch=True)
        
        # Log each metric type per `cath_degree`
        for metric_key, log_name in metric_types.items():
            for cath_degree, value in metrics[metric_key].items():
                self.log(f'{log_name}_degree_{cath_degree}', value, on_epoch=True)
        
        # Log counts
        self.log("total_count", metrics['total_count'], on_epoch=True)
        for cath_degree, count in metrics['counts_per_degree'].items():
            self.log(f'count_degree_{cath_degree}', count, on_epoch=True)
        
        log_histograms(
        logger=self.logger,
        cath_degrees=metrics['cath_degree_per_sample'],
        pocket_rmsds=metrics['pocket_rmsd_per_sample'],
        epoch=self.current_epoch,
        bins=20  # You can adjust the number of bins as needed
    )

        self.logger.experiment.log_image(image_data=generate_and_log_scatter_plot(metrics), name="Pocket RMSD")
            
        self._metrics.reset()
    
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        batch_size = batch['tar_embedding'].shape[0]
        for loss_name, value in outputs['loss_dict'].items():
            self.log(f'valid_{loss_name}_loss', value, batch_size=batch_size, prog_bar=False, on_epoch=True)
        self.log(f'valid_loss', outputs['loss'], batch_size=batch_size, prog_bar=False, on_epoch=True)

    def create_correspondences_matrix(self, batch, tar_embedding: torch.Tensor, src_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        ret_dict = {}
        tar_embedding, src_embedding  = self._input_tar_block(tar_embedding), self._input_src_block(src_embedding)

        l2_embedding = torch.sqrt(torch.sum((src_embedding.unsqueeze(1) - tar_embedding.unsqueeze(2)) ** 2, dim=-1))
        if self._use_scalar_layer:
            tar_scalar = self._tar_linear(tar_embedding).squeeze(-1) 
            src_scalar = self._src_linear(src_embedding).squeeze(-1)

            if self._compute_pocket_importance and not self.training:
                tar_pocket_mean, tar_non_pocket_mean = compute_mean_scalar_pocket_values(tar_scalar, batch['tar_residue_indices'], batch['tar_pocket_mask'], batch['tar_mask'].sum(1))
                src_pocket_mean, src_non_pocket_mean = compute_mean_scalar_pocket_values(src_scalar, batch['src_residue_indices'], batch['src_pocket_mask'], batch['src_mask'].sum(1))
                ret_dict.update({"tar_pocket_scalar_mean": tar_pocket_mean, "tar_non_pocket_scalar_mean": tar_non_pocket_mean, "src_pocket_scalar_mean": src_pocket_mean, "src_non_pocket_scalar_mean": src_non_pocket_mean})
                
            tar_matrix = tar_scalar.unsqueeze(-1).expand_as(l2_embedding)  # Shape [B, N_tar, N_src]
            src_matrix = src_scalar.unsqueeze(1).expand_as(l2_embedding)  # Shape [B, N_tar, N_src]

            ret_dict.update({"l2_embedding" :l2_embedding + tar_matrix + src_matrix, "tar_embedding": tar_embedding, "src_embedding": src_embedding})
        else:
            ret_dict.update({"l2_embedding" :l2_embedding, "tar_embedding": tar_embedding, "src_embedding": src_embedding})
        return ret_dict
    
    def training_step(self, batch: dict[torch.Tensor], batch_idx: int):
        batch = move_batch_to_device(batch, self.device)
        batch_size = batch['tar_embedding'].shape[0]

        embeddings_dict = self.create_correspondences_matrix(batch, batch['tar_embedding'], batch['src_embedding'])

        combined_mask = self.create_2d_mask(batch)
        l2_embedding = embeddings_dict['l2_embedding'] * combined_mask
        l2_embedding = l2_embedding.masked_fill(~combined_mask, float('inf'))
        gamma_0 = self._mask_and_normalize_matrix(l2_embedding, batch['src_mask'], batch['tar_mask'], embeddings_dict['src_embedding'], embeddings_dict['tar_embedding']) * combined_mask

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
        loss_dict: dict[str, torch.Tensor] = self._pocket_loss(batch, R_total, t_total)
        loss = loss_dict['pocket_rmsd']
        if self._transformation_loss:
            loss_dict.update(self._transformation_loss(batch, R_total, t_total))
            loss = self._alpha_loss * loss_dict['pocket_rmsd'] + (1-self._alpha_loss) * loss_dict['transformation']
            # loss_dict = loss_dict | transformation_loss
        loss_dict['loss'] = loss
        return loss, loss_dict

    def validation_step(self, batch, batch_idx):
        batch = move_batch_to_device(batch, self.device)
        batch_size = batch['tar_embedding'].shape[0]

        embeddings_dict = self.create_correspondences_matrix(batch, batch['tar_embedding'], batch['src_embedding'])

        combined_mask = self.create_2d_mask(batch)
        l2_embedding = embeddings_dict['l2_embedding'] * combined_mask
        l2_embedding = l2_embedding.masked_fill(~combined_mask, float('inf'))
        gamma_0 = self._mask_and_normalize_matrix(l2_embedding, batch['src_mask'], batch['tar_mask'], embeddings_dict['src_embedding'], embeddings_dict['tar_embedding']) * combined_mask
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
        
        outputs = {'loss': loss , 'loss_dict': loss_dict, 'pred_R': R_total, 'pred_t': t_total, 'pred_first_R': all_R[0], 'pred_first_t': all_t[0], "embeddings_dict": embeddings_dict}
        self._metrics.update(batch, outputs)
        return outputs

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self._lr)
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        
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
    import logging
    import yaml
    from torch.utils.data import DataLoader
    from pytorch_lightning.loggers import CometLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
    # from pytorch_lightning.profilers import AdvancedProfiler, SimpleProfiler


    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", "learnable_softbbs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, start_time + ".log"), level=logging.INFO, format='%(message)s')

    logger = logging.getLogger(__name__)
    torch.manual_seed(42)


    with open('configs/learnable_soft_bb.yaml') as f:
        config = yaml.safe_load(f)

    train_dataset = build_object(config['dataset']['train'], 'datasets')
    valid_dataset = build_object(config['dataset']['validation'], 'datasets')
    train_loader  = DataLoader(train_dataset, batch_size=config['dataloader']['train_batch_size'], collate_fn=custom_collate_fn, num_workers=config['dataloader']['n_workers'])
    val_loader  = DataLoader(valid_dataset, batch_size=config['dataloader']['valid_batch_size'], collate_fn=custom_collate_fn, num_workers=config['dataloader']['n_workers'])

    model = build_object(config['model'], 'models')

    comet_logger = CometLogger(
        api_key="9ydBzigeK75Z6RhAiX63xGdsg",
        workspace="hagairavid18",
        project_name="pocket_aligner",
        experiment_name=config['trainer']['exp_name']
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{config['trainer']['exp_name']}",
        save_top_k=-1,
        every_n_epochs=1, 
    )

    trainer = L.Trainer(logger=comet_logger,
                        # profiler = AdvancedProfiler(filename="profile_results_cloud_noprotein.txt", dirpath='.') if config['trainer']['profiler'] == True else None,
                        max_epochs=config['trainer']['max_epochs'], 
                        check_val_every_n_epoch=config['trainer']['check_val_every_n_epoch'],
                        # precision=16,
                        callbacks=[checkpoint_callback], 
                        gradient_clip_val= config['trainer']['gradient_clipping'], 
                        log_every_n_steps=100, 
                        accelerator= 'gpu' if torch.cuda.is_available() else 'cpu')
        
    trainer.logger.log_hyperparams(flatten_dict(config))

    if config['trainer']['validate_only']:
        trainer.validate(model, val_loader, ckpt_path=config['trainer']['ckpt_path'])
    else:
        trainer.fit(model, train_loader, val_dataloaders=val_loader, ckpt_path=config['trainer']['ckpt_path'])

        