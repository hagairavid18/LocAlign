from abc import ABC, abstractmethod
import pandas as pd
import os
from typing import Any
import torch
import lightning as L
import torch.optim as optim

from metrics import PocketRMSD
from models.utils.misc import build_object

torch.set_float32_matmul_precision('medium')


class SoftBBBase(L.LightningModule, ABC):
    def __init__(self, loss: dict[str, Any], optimizer: dict[str, Any], max_iter: int = 5, use_atom_level: bool = False) -> None:
        """
        Base class for algorithms implementing the SoftBB algorithm. Generates a soft correspondence matrix between two sets of embeddings and computes the optimal transformation between them.
        Iterate over the optimal transformation and the correspondence matrix to minimize the pocket RMSD loss function.

        Args:
            loss (dict[str, Any]): loss functions to be used in the model.
            optimizer (dict[str, Any]): optimizer configuration.
            max_iter (int, optional): Since the process is iterative, we define max iterations. Defaults to 5.
            use_atom_level (bool, optional): True if inputs are atom, false for residues. Defaults to False.
        """        
        super().__init__()
        self._pocket_loss = build_object(loss['pocket'], 'losses')
        self._transformation_loss = build_object(loss['transformation'], 'losses')
        self._use_transformation_loss = loss['use_transformation']
        self._alpha_loss = 0.5
        self._metrics = PocketRMSD()
        self._min_diff = 0.05
        self._max_iter = max_iter
        self._lr = optimizer['args']['learning_rate']
        self._use_atom_level = use_atom_level
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        batch_size = batch['tar_embedding'].shape[0]
        for loss_name, value in outputs['loss_dict'].items():
            self.log(f'train_{loss_name}_loss', value, batch_size=batch_size, prog_bar=False, on_step=True, on_epoch=True)
        self.log(f'train_loss', outputs['loss'], batch_size=batch_size, prog_bar=False, on_step=True, on_epoch=True)
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
      
        self._metrics.reset()
    
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        batch_size = batch['tar_embedding'].shape[0]
        for loss_name, value in outputs['loss_dict'].items():
            self.log(f'valid_{loss_name}_loss', value, batch_size=batch_size, prog_bar=False, on_epoch=True)
        self.log(f'valid_loss', outputs['loss'], batch_size=batch_size, prog_bar=False, on_epoch=True)
    
    def compute_correspondences(self, batch, combined_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute a soft correspondences matrix  between the source and target embeddings.

        Args:
            batch (_type_): contains the source and target embeddings.
            combined_mask (torch.Tensor): 2D mask for the combined mask.

        Returns:
            _type_: _description_
        """        
        
        embeddings_dict = self.create_correspondences_matrix(batch, batch['tar_embedding'], batch['src_embedding'])
        print(f"l2_embedding mean: {embeddings_dict['l2_embedding'].mean()}, l2_embedding std: {embeddings_dict['l2_embedding'].std()}")
        l2_embedding = embeddings_dict['l2_embedding'] * combined_mask
        print(f"l2_embedding mean after mask: {l2_embedding.mean()}, l2_embedding std after mask: {l2_embedding.std()}")
        l2_embedding = l2_embedding.masked_fill(~combined_mask, float('inf'))
        return embeddings_dict, l2_embedding

    
    @abstractmethod
    def create_correspondences_matrix(self) -> dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def training_step(self):
       pass
    
    @abstractmethod
    def validation_step(self):
        pass

    def _compute_loss(self, batch, R_total, t_total):
        loss_dict: dict[str, torch.Tensor] = self._pocket_loss(batch, R_total, t_total)
        loss = loss_dict['non_linear_pocket_rmsd']
        loss_dict.update(self._transformation_loss(batch, R_total, t_total))
        if self._use_transformation_loss:
            loss = self._alpha_loss * loss_dict['non_linear_pocket_rmsd'] + (1-self._alpha_loss) * loss_dict['transformation']
        loss_dict['loss'] = loss
        return loss, loss_dict


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
    