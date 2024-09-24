
from datetime import datetime
import os
import torch
from torch.utils.data import DataLoader
import lightning as L
from scipy.spatial.transform import Rotation

from utils.kabsch import weighted_kabsch_torch
from models.utils.collate import custom_collate_fn
from utils.deepbbs_utils import *
from utils.plots import plot_transformed_point_clouds


class SoftBB(L.LightningModule):
    def __init__(self):
        super().__init__()
        self._validation_outputs = {}
        self.validation_step_outputs = []
        self.alpha_factor = 4
        self.eps = 0.00001
        self._min_diff = 0.05
        self._max_iter = 10
    
   
    def training_step(self, batch: dict[torch.Tensor], batch_idx: int):
        pass
     
    def _mask_and_normalize_matrix(self, distance_matrix: torch.Tensor, src_mask: torch.Tensor, tar_mask: torch.Tensor, src_embedding: torch.Tensor, tar_embedding) -> list[tuple]:
        batch_size = distance_matrix.shape[0]
        t = torch.tensor([guess_best_alpha_torch(src_embedding[i,:][src_mask[i]], dim_num=tar_embedding.shape[-1], transpose=False) for i in range(batch_size)])
        # R = torch.cat([softargmin_rows_torch(distance_matrix[i], t[i]) for i in range(batch_size)])
        R = torch.stack([softargmin_rows_torch(distance_matrix[i], t[i]) for i in range(batch_size)], dim=0)
        t = torch.tensor([guess_best_alpha_torch(tar_embedding[i,:][tar_mask[i]], dim_num=tar_embedding.shape[-1], transpose=False) for i in range(batch_size)])
        # C = softargmin_rows_torch(torch.transpose(distance_matrix, dim0=1, dim1=2), t)
        C = torch.stack([softargmin_rows_torch(torch.transpose(distance_matrix, dim0=1, dim1=2)[i], t[i]) for i in range(batch_size)], dim=0)
        C = torch.transpose(C, dim0=1, dim1=2)
        B = torch.mul(R, C)
      
        return B
    
    def validation_step(self, batch, batch_idx):
        print(batch['metadata'][0]['index'])
        tar_embedding, src_embedding  = batch['tar_embedding'], batch['src_embedding']
        tar_expanded = tar_embedding.unsqueeze(2)  # Shape: (BATCH, A, 1, C)
        src_expanded = src_embedding.unsqueeze(1)  # Shape: (BATCH, 1, B, C)
        batch_size = batch['tar_embedding'].shape[0]
        mask_dim1_expanded = batch['tar_mask'].unsqueeze(2).expand(batch_size, -1, 1000) 
        mask_dim2_expanded = batch['src_mask'].unsqueeze(1).expand(batch_size, 1000, -1)  
        combined_mask = mask_dim1_expanded * mask_dim2_expanded 
        d_0  = (1.24*(torch.Tensor(batch['max_length'])-15)**(1/3) -1.8)

        # Calculate squared differences
        l2_embedding = torch.sqrt(torch.sum((src_expanded - tar_expanded) ** 2, dim=-1))  # Shape: (BATCH, A, B)
        l2_embedding = l2_embedding * combined_mask
        l2_embedding = l2_embedding.masked_fill(~combined_mask, float('inf'))
        gamma_0 = self._mask_and_normalize_matrix(l2_embedding, batch['src_mask'], batch['tar_mask'], src_embedding, tar_embedding) * combined_mask
        # gamma = self._mask_and_normalize_matrix(l2_embedding, batch['src_mask'], batch['tar_mask'], torch.zeros_like(src_embedding), torch.zeros_like(tar_embedding)) * combined_mask
        gamma = gamma_0.clone()
        src_coordinates = batch['src_coordinates']
        all_R, all_t = [], []
        not_converged = True
        iter_num = 1
        while(not_converged):

            R_gamma, t_gamma, rmsd_gamma, rmsd_per_corr_gamma = weighted_kabsch_torch(src_coordinates,  batch['tar_coordinates'], gamma.float(),  batch['src_mask'],  batch['tar_mask'])
            all_R.append(R_gamma)
            all_t.append(t_gamma)
            rotation = Rotation.from_matrix(R_gamma)
            euler_angles = rotation.as_euler('xyz', degrees=False)
            diff = np.mean(np.abs(euler_angles))
            print(diff)
            if diff < self._min_diff or iter_num >= self._max_iter:
                not_converged = False
            src_coordinates = (torch.matmul(src_coordinates, R_gamma) + t_gamma.unsqueeze(1)) * batch['src_mask'].unsqueeze(-1).expand_as(batch['src_coordinates'])
            
            src_tgt_euc_dist = cdist_torch(batch['tar_coordinates'], src_coordinates, 3)
            gamma = (gamma_0 / (( 1 + (src_tgt_euc_dist / d_0[:, None, None])**2)**2))* combined_mask
            iter_num +=1        
           
        R_total = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)  # Initial rotation matrix (B x 3 x 3)
        t_total = torch.zeros(batch_size, 3)  # Initial translation vector (B x 3)

        for R, t in zip(all_R, all_t):
            R_total = R_total @ R
            t_total = torch.bmm(t_total.unsqueeze(1), R).squeeze(1) + t
                
        transformations_to_plot = {"GT" : (batch['gt_R'], batch['gt_t'][:,:3]),
                                    "TMalign": (torch.Tensor(batch['metadata'][0]['TMaligner_rotations']), torch.Tensor(batch['metadata'][0]['TMaligner_translations'])),
                                    "iterative_SoftBBS" :(R_total, t_total)}
        plot_transformed_point_clouds(batch, transformations_to_plot,  compose = False)

        self._validation_outputs[batch['metadata'][0]['index']] = {
            'svd_weighted_R': all_R[0],
            'svd_weighted_t': all_t[0],
            'iterative_svd_weighted_R': R_total,
            'iterative_svd_weighted_t': t_total,
            }

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) > 0:

            avg_val_loss = torch.stack(self.validation_step_outputs).mean()
            print(avg_val_loss)
            self.log('avg_val_loss', avg_val_loss, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    @property
    def automatic_optimization(self):
        return True
    

if __name__ == "__main__":
    from datasets import ScannetDataset
    from utils.misc import save_results_to_csv
    import logging
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", "bbs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, start_time + ".log"), level=logging.INFO, format='%(message)s')

    logger = logging.getLogger(__name__)
    train = True
    # data_path = '/home/iscb/wolfson/hagairavid/ligand_alligner/results/baseline_results/2024-07-11_10-55-38_57.csv'
    # data_path = '/home/iscb/wolfson/hagairavid/ligand_alligner/results/baseline_results/2024-07-17_16-01-08_3000.csv'
    data_path = 'results/hard_bbs_results/2024-08-14_22-39-17_2000.csv'
    # data_path = 'results/hard_bbs_results/2024-08-14_21-50-16_500.csv'
    base_data_path = os.path.join('/home/iscb/wolfson/hagairavid/ligands')
    
    valid_dataset = ScannetDataset(data_path, base_data_path, 2000)
    val_loader  = DataLoader(valid_dataset, batch_size=1, collate_fn=custom_collate_fn, num_workers=0)
    model = SoftBB()

    trainer = L.Trainer(max_epochs=5)
    # trainer.validate(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.validate(model,  dataloaders=val_loader)
    output_df = valid_dataset._df.copy()

    # output_df['HardBBS_rotations'] = [[] for _ in range(len(output_df))]
    # output_df['HardBBS_translations'] = [[] for _ in range(len(output_df))]
    output_df['iterative_SoftBBS_rotations'] = [[] for _ in range(len(output_df))]
    output_df['iterative_SoftBBS_translations'] = [[] for _ in range(len(output_df))]
    output_df['SoftBBS_rotations'] = [[] for _ in range(len(output_df))]
    output_df['SoftBBS_translations'] = [[] for _ in range(len(output_df))]
    for row_idx, values in model._validation_outputs.items():
        # output_df.at[row_idx,'HardBBS_translations'] = values['svd_t'].tolist()
        # output_df.at[row_idx,'HardBBS_rotations'] = values['svd_R'].tolist()
        output_df.at[row_idx,'iterative_SoftBBS_translations'] = values['iterative_svd_weighted_t'].tolist()
        output_df.at[row_idx,'iterative_SoftBBS_rotations'] = values['iterative_svd_weighted_R'].tolist()
        output_df.at[row_idx,'SoftBBS_translations'] = values['svd_weighted_t'].tolist()
        output_df.at[row_idx,'SoftBBS_rotations'] = values['svd_weighted_R'].tolist()
    
    save_results_to_csv(output_df, start_time, base_dir="iterative_soft_bbs_results")