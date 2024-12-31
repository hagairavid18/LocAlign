
from datetime import datetime
import os
import torch
import lightning as L

from models.utils.collate import custom_collate_fn, move_batch_to_device
from metrics import PocketRMSD
from models.utils.misc import build_object, flatten_dict
from models.utils.plots import generate_and_log_scatter_plot


class TMaligner(L.LightningModule):
    def __init__(self):
        super().__init__()
        self._metrics = PocketRMSD()
        
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
                    'bbr': pair_info[3],
                    'CATH Degree': cath_degree,
                    'Pocket RMSD': pocket_rmsd.item()
                })
        
        
        if protein_rmsd_data:
            import pandas as pd
            dir_path = os.path.join("results", "validation_results", self.logger.experiment.get_name())
            os.makedirs(dir_path, exist_ok=True)
            df = pd.DataFrame(protein_rmsd_data)
            df.to_csv(os.path.join(dir_path, f"Protein_RMSD_Results_{self.current_epoch}.csv"))
            self.logger.experiment.log_table(f"Protein_RMSD_Results_{self.current_epoch}.csv", df)
        
        
        self._metrics.reset()

    def validation_step(self, batch, batch_idx):
        batch = move_batch_to_device(batch, self.device)
        assert len(batch['metadata']) == 1
        R = torch.Tensor(batch['metadata'][0]['TMaligner_rotations'])
        t = torch.Tensor(batch['metadata'][0]['TMaligner_translations'])        
        outputs = {'pred_R': R, 'pred_t': t}
        self._metrics.update(batch, outputs)
        return outputs


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
        