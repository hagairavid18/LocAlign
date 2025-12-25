from abc import ABC
import pandas as pd
import os
from typing import Any
import torch
import torch
torch.set_float32_matmul_precision('medium')  # or 'high'

import lightning as L
import torch.optim as optim

from models.utils.misc import build_object


class SoftBBBase(L.LightningModule, ABC):
    def __init__(
            self, 
            loss: dict[str, Any] | None, 
            metric: dict[str, Any] | None,
            optimizer: dict[str, Any] | None, 
            ) -> None:
        """
        Base class for algorithms implementing the SoftBB algorithm. Generates a soft correspondence matrix between two sets of 
        embeddings and computes the optimal transformation between them.
        Iterate over the optimal transformation and the correspondence matrix to minimize the pocket RMSD loss function.

        Args:
            loss (dict[str, Any]): loss functions to be used in the model.
            optimizer (dict[str, Any]): optimizer configuration.
        """        
        super().__init__()
        self._loss =  build_object(loss, 'losses')
        self._metrics = build_object(metric, 'metrics')
        self._lr = optimizer['args']['learning_rate'] if optimizer is not None else 0.001
        self._scheduler_config = optimizer['args'].pop('scheduler', None) if optimizer is not None else None
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        batch_size = batch['tar_pretrained_embeddings'].shape[0]

        loss_logs = {f"train_{k}_loss": v for k, v in outputs['loss_dict'].items()}
        loss_logs["train_loss"] = outputs['loss']

        self.log_dict(loss_logs, batch_size=batch_size, prog_bar=False, on_step=True, on_epoch=True)

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=True, on_epoch=True, logger=True)
        self._loss.update_lambda(self.global_step, self.total_steps)
        self.log('corr_lambda', self._loss._quality_loss.corr_lambda.item(), on_step=True, on_epoch=True, logger=True)
    
    def on_validation_epoch_end(self):
        metrics = self._metrics.compute()
        
        metric_types = {
            'ligand_rmsd': 'valid_ligand_rmsd',
            'ligand_rmsd_below_4_proportion_per_degree': 'ligand_rmsd_below_4',
            'weighted_same_type_per_degree': 'atom_type_same_fraction',
            'random_baseline_per_degree': 'random_baseline_per_degree',
        }
        
        # Log total metrics
        for metric_key, log_name in metric_types.items():
            total_value = sum(torch.tensor(list(metrics[metric_key].values())) * torch.tensor(list(metrics['counts_per_degree'].values()))) / metrics['total_count']
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
        
        for cath_degree in range(0, 9):
            pair_infos = metrics['pair_infos_per_degree'][cath_degree]
            ligand_rmsd_values = metrics['ligand_rmsd_per_degree_protein'][cath_degree]
            embedding_similarity_values = metrics['embedding_similarity_per_degree_protein'][cath_degree]
            corr_rmsd_values = metrics['corr_rmsd_per_degree_protein'][cath_degree]
            gap_values = metrics['gap_per_degree_protein'][cath_degree]
            atom_type_values = metrics['weighted_same_type_per_degree_protein'][cath_degree]
            radius_values = metrics['radius_per_degree_protein'][cath_degree]
            keys_to_keep = ['ligand_id', 'tar_protein', 'tar_chain', 'src_protein', 'src_chain', 'cath_degree', 'src_ligand_n_atoms', 'tar_ligand_n_atoms']
            for pair_info, ligand_rmsd, embedding_similarity, corr_rmsd, gap, atom_val, radius in zip(pair_infos, ligand_rmsd_values, embedding_similarity_values, corr_rmsd_values, gap_values, atom_type_values, radius_values):
                protein_rmsd_data.append({
                    **{k: pair_info[k] for k in keys_to_keep},
                    'src_ligand': pair_info.get('ligand_id', ''),
                    'tar_ligand': pair_info.get('ligand_id', ''),
                    'ligand_rmsd': ligand_rmsd.item(),
                    'embedding_similarity': embedding_similarity.item(),
                    'corr_rmsd': corr_rmsd.item(),
                    'entropy': gap.item(),
                    'atom_type_fraction': atom_val,
                    'radius_of_gyration': radius.item(),
                })
        
        if protein_rmsd_data and hasattr(self.logger.experiment, 'get_name'):
            dir_path = os.path.join("results", "validation_results", self.logger.experiment.get_name())
            os.makedirs(dir_path, exist_ok=True)
            df = pd.DataFrame(protein_rmsd_data)
            df.to_csv(os.path.join(dir_path, f"per_sample_results_{self.current_epoch}.csv"))
            self.logger.experiment.log_table(f"per_sample_results_{self.current_epoch}.csv", df)
      
        self._metrics.reset()
    
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        if 'loss_dict' not in outputs:
            return
        outputs['loss_dict'].pop('per_sample', None)
        batch_size = batch['tar_pretrained_embeddings'].shape[0]
        for loss_name, value in outputs['loss_dict'].items():
            self.log(f'valid_{loss_name}_loss', value, batch_size=batch_size, prog_bar=False, on_epoch=True)
        self.log(f'valid_loss', outputs['loss'], batch_size=batch_size, prog_bar=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self._lr, weight_decay=1e-4, fused=True)        
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
    