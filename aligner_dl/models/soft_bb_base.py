from abc import ABC
import pandas as pd
import os
from typing import Any
import torch
import torch
torch.set_float32_matmul_precision('medium')  # or 'high'

import lightning as L
import torch.optim as optim

from metrics import PocketRMSD
from models.utils.plots import generate_and_log_scatter_plot
from models.utils.misc import build_object


class SoftBBBase(L.LightningModule, ABC):
    def __init__(
            self, 
            loss: dict[str, Any] | None, 
            optimizer: dict[str, Any] | None, 
            max_iter: int = 5, 
            n_iter_train: int = 2, 
            plot_dir : str | None = None
            ) -> None:
        """
        Base class for algorithms implementing the SoftBB algorithm. Generates a soft correspondence matrix between two sets of 
        embeddings and computes the optimal transformation between them.
        Iterate over the optimal transformation and the correspondence matrix to minimize the pocket RMSD loss function.

        Args:
            loss (dict[str, Any]): loss functions to be used in the model.
            optimizer (dict[str, Any]): optimizer configuration.
            max_iter (int, optional): Since the process is iterative, we define max iterations. Defaults to 5.
            n_iter_train (int, optional): Number of weighted kabsch iterations to train. Defaults to 2.
            plot_dir (str | None, optional): Directory to save plots. Defaults to None.
        """        
        super().__init__()
        self._pocket_loss = build_object(loss['pocket'], 'losses') if loss is not None else None
        self._transformation_loss = build_object(loss['transformation'], 'losses') if loss is not None else None
        self._ligand_loss = build_object(loss['ligand'], 'losses') if loss is not None else None
        self._use_transformation_loss = loss['use_transformation'] if loss is not None else False
        self._alpha_loss = 0.5
        self._metrics = PocketRMSD()
        self._max_iter = max_iter
        self._n_iter_train = n_iter_train
        self._lr = optimizer['args']['learning_rate'] if optimizer is not None else 0.001
        self._scheduler_config = optimizer['args'].pop('scheduler', None) if optimizer is not None else None
        self._plot = False
        if plot_dir is not None:
            os.makedirs(plot_dir, exist_ok=True)
            self._plot = True
            self._plot_dir = plot_dir
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        batch_size = batch['tar_embedding'].shape[0]

        loss_logs = {f"train_{k}_loss": v for k, v in outputs['loss_dict'].items()}
        loss_logs["train_loss"] = outputs['loss']

        self.log_dict(loss_logs, batch_size=batch_size, prog_bar=False, on_step=True, on_epoch=True)

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=True, on_epoch=True, logger=True)
    
    def on_validation_epoch_end(self):
        metrics = self._metrics.compute()
        
        metric_types = {
            'pocket_rmsd': 'valid_pocket_rmsd',
            'pocket_rmsd_iter0': 'valid_pocket_rmsd_iter0',
            'ligand_rmsd': 'valid_ligand_rmsd',
            'ligand_rmsd_iter0': 'valid_ligand_rmsd_iter0',
            'src_pocket_embeddings_scalar': 'src_pocket_embeddings_scalar',
            'tar_pocket_embeddings_scalar': 'tar_pocket_embeddings_scalar',
            'src_non_pocket_embeddings_scalar': 'src_non_pocket_embeddings_scalar',
            'tar_non_pocket_embeddings_scalar': 'tar_non_pocket_embeddings_scalar',
            'rmsd_below_4_proportion_per_degree': 'rmsd_below_4',
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
        
        # Log protein names and pocket_rmsd per degree in a table
        protein_rmsd_data = []
        
        for cath_degree in range(1, 9):
            pair_infos = metrics['pair_infos_per_degree'][cath_degree]
            pocket_rmsd_values = metrics['pocket_rmsd_per_degree_protein'][cath_degree]
            
            for pair_info, pocket_rmsd in zip(pair_infos, pocket_rmsd_values):
                protein_rmsd_data.append({
                    'ligand': pair_info[0],
                    'src protein': pair_info[1],
                    'tar protein': pair_info[2],
                    'CATH Degree': cath_degree,
                    'Pocket RMSD': pocket_rmsd.item()
                })
        
        if protein_rmsd_data and hasattr(self.logger.experiment, 'get_name'):
            dir_path = os.path.join("results", "validation_results", self.logger.experiment.get_name())
            os.makedirs(dir_path, exist_ok=True)
            df = pd.DataFrame(protein_rmsd_data)
            df.to_csv(os.path.join(dir_path, f"Protein_RMSD_Results_{self.current_epoch}.csv"))
            self.logger.experiment.log_table(f"Protein_RMSD_Results_{self.current_epoch}.csv", df)
            self.logger.experiment.log_image(generate_and_log_scatter_plot(metrics))
      
        self._metrics.reset()
    
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        if 'loss_dict' not in outputs:
            return
        batch_size = batch['tar_embedding'].shape[0]
        for loss_name, value in outputs['loss_dict'].items():
            self.log(f'valid_{loss_name}_loss', value, batch_size=batch_size, prog_bar=False, on_epoch=True)
        self.log(f'valid_loss', outputs['loss'], batch_size=batch_size, prog_bar=False, on_epoch=True)

        # if batch_idx % 10 == 0 and self._plot:
        #     plot_transformed_point_clouds_interactive(self.logger, batch, outputs['transformation_dict'], epoch=self.current_epoch, step=batch_idx)

    def _compute_loss(self, batch, R_total, t_total):
        loss_dict: dict[str, torch.Tensor] = self._pocket_loss(batch, R_total, t_total)
        loss = loss_dict['pocket_rmsd']
        loss_dict.update(self._transformation_loss(batch, R_total, t_total))
        loss_dict.update(self._ligand_loss(batch, R_total, t_total))
        # loss = loss_dict['ligand_rmsd']
        if self._use_transformation_loss:
            loss = self._alpha_loss * loss_dict['pocket_rmsd'] + (1-self._alpha_loss) * loss_dict['transformation']
        loss_dict['loss'] = loss
        return loss, loss_dict

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self._lr, weight_decay=1e-4, fused=False)        
        # optimizer = optim.Adam(self.parameters(), lr=self._lr)        
        if self._scheduler_config is not None:
            self._scheduler_config['args']['optimizer'] = optimizer
            interval = self._scheduler_config['args'].pop('interval', 'step')
            scheduler = build_object(self._scheduler_config, "torch.optim.lr_scheduler")
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
            interval = 'epoch'
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'valid_loss',  # Monitors validation loss or another metric
                'interval': interval,      # Frequency to update the scheduler ('epoch' or 'step')
                'frequency': 1,           # Frequency of calling the scheduler
            }
        }
    