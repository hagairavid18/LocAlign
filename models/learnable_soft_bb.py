
from datetime import datetime
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L
import pytorch_lightning as pl
import GPUtil


from utils.kabsch import weighted_kabsch_torch
from models.utils.collate import custom_collate_fn, move_batch_to_device
from utils.deepbbs_utils import *
from utils.transformation import rotation_matrix_to_euler_angles
from losses import RTLoss

from utils.plots import plot_transformed_point_clouds

torch.set_float32_matmul_precision('medium')


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.BatchNorm1d(output_dim)  # Using BatchNorm1d for normalization
        self.activation = nn.ReLU()  # Using ReLU for activation

    def forward(self, x):
        # x is of shape B x N x d
        B, N, d = x.shape
        x = self.linear(x.view(-1, d))  # Apply linear layer
        x = self.norm(x)  # Apply normalization
        x = self.activation(x)  # Apply activation
        return x.view(B, N, -1)  # Reshape back to B x N x k
    

class FeatureCoordinateBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeatureCoordinateBlock, self).__init__()
        
        # First linear layer to combine input features and coordinates
        self.fc1 = nn.Linear(input_dim + 3, hidden_dim)  # Combining D (input feature) and 3 (coordinates)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

        # Second linear layer to further process the combined features
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.batch_norm2 = nn.BatchNorm1d(output_dim)

    def forward(self, features, coordinates):
        # Concatenate features and coordinates along the last dimension
        combined_input = torch.cat([features, coordinates], dim=-1)  # Shape: (B, N, D+3)
        
        # Apply the first linear transformation
        B, N, _ = combined_input.shape
        combined_output = self.fc1(combined_input.view(B * N, -1))  # Shape: (B*N, hidden_dim)
        combined_output = self.batch_norm1(combined_output)  # Batch normalization
        combined_output = self.relu(combined_output)  # ReLU activation
        
        # Apply the second linear transformation
        combined_output = self.fc2(combined_output)  # Shape: (B*N, output_dim)
        combined_output = self.batch_norm2(combined_output)  # Batch normalization
        combined_output = self.relu(combined_output)  # ReLU activation

        # Reshape back to original shape
        combined_output = combined_output.view(B, N, -1)
        
        return combined_output
class SoftBB(L.LightningModule):
    def __init__(self):
        super().__init__()
        self._validation_outputs = {}
        self._inout_tar_block = FeatureCoordinateBlock(128 ,128, 128)
        self._inout_src_block = FeatureCoordinateBlock(128 ,128, 128)
        self._loss = RTLoss()
        self.validation_step_outputs = []
        self.eps = 0.00001
        self._min_diff = 0.05
        self._max_iter = 1
    
    def get_d0(self, max_length) -> torch.Tensor:
        return 1.24*(torch.tensor(max_length, device=self.device) - 15)**(1/3) -1.8
    
    def _compose_transformations(self, batch_size, rotations, translations) -> tuple[torch.Tensor, torch.Tensor]:
        R_total = torch.eye(3, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)  # Initial rotation matrix (B x 3 x 3)
        t_total = torch.zeros(batch_size, 3, device=self.device)  # Initial translation vector (B x 3)

        for i, (R, t) in enumerate(zip(rotations, translations)):
            R_total = R_total @ R
            t_total = torch.bmm(t_total.unsqueeze(1), R).squeeze(1) + t

        return R_total, t_total
    
    def create_2d_mask(self, batch) -> torch.Tensor:
        batch_size = batch['tar_embedding'].shape[0]
        mask_dim1_expanded = batch['tar_mask'].unsqueeze(2).expand(batch_size, -1, 1000) 
        mask_dim2_expanded = batch['src_mask'].unsqueeze(1).expand(batch_size, 1000, -1)  
        combined_mask = mask_dim1_expanded * mask_dim2_expanded
        return combined_mask

    def on_train_batch_end(self, outputs, batch, batch_idx):
        batch_size = batch['tar_embedding'].shape[0]
        self.log('train_loss', outputs['loss_dict']['loss'], batch_size=batch_size, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_rotation_loss', outputs['loss_dict']['rot_loss'], batch_size=batch_size, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_translation_loss', outputs['loss_dict']['tran_loss'], batch_size=batch_size, prog_bar=True, on_step=True, on_epoch=True)
        transformations_to_plot = {"GT" : (batch['gt_R'], batch['gt_t'][:,:3]),
                                    "TMalign": (torch.Tensor(batch['metadata'][0]['TMaligner_rotations']), torch.Tensor(batch['metadata'][0]['TMaligner_translations'])),
                                    "iterative_SoftBBS" :(outputs['pred_R'].clone().detach(), outputs['pred_t'].clone().detach())}
        if batch_idx % 10 == 0:
            logger.experiment.add_image("Transformed Point Clouds", plot_transformed_point_clouds(batch, transformations_to_plot,  compose = False), self.global_step)
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                self.logger.experiment.add_scalar(f"GPU_{gpu.id}/Memory_Usage_MB", gpu.memoryUsed / 1024, self.global_step)
                self.logger.experiment.add_scalar(f"GPU_{gpu.id}/GPU_Utilization", gpu.load * 100, self.global_step)
    
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        batch_size = batch['tar_embedding'].shape[0]
        self.log('valid_loss', outputs['loss_dict']['loss'], batch_size=batch_size, prog_bar=True, on_epoch=True)
        self.log('valid_rotation_loss', outputs['loss_dict']['rot_loss'], batch_size=batch_size, prog_bar=True, on_epoch=True)
        self.log('valid_translation_loss', outputs['loss_dict']['tran_loss'], batch_size=batch_size, prog_bar=True, on_epoch=True)
        transformations_to_plot = {"GT" : (batch['gt_R'], batch['gt_t'][:,:3]),
                                    "TMalign": (torch.Tensor(batch['metadata'][0]['TMaligner_rotations']), torch.Tensor(batch['metadata'][0]['TMaligner_translations'])),
                                    "iterative_SoftBBS" :(outputs['pred_R'].clone().detach(), outputs['pred_t'].clone().detach())}
        if batch_idx % 50 == 0:
            logger.experiment.add_image("Transformed Point Clouds", plot_transformed_point_clouds(batch, transformations_to_plot,  compose = False), self.global_step)
    
    def training_step(self, batch: dict[torch.Tensor], batch_idx: int):
        batch = move_batch_to_device(batch, self.device)
        batch_size = batch['tar_embedding'].shape[0]
        tar_embedding, src_embedding  = self._inout_tar_block(batch['tar_embedding'], batch['tar_coordinates'] ), self._inout_src_block(batch['src_embedding'], batch['src_coordinates'])
        combined_mask = self.create_2d_mask(batch)

        l2_embedding = torch.sqrt(torch.sum((src_embedding.unsqueeze(1) - tar_embedding.unsqueeze(2)) ** 2, dim=-1))
        l2_embedding = l2_embedding * combined_mask
        l2_embedding = l2_embedding.masked_fill(~combined_mask, float('inf'))
        gamma_0 = self._mask_and_normalize_matrix(l2_embedding, batch['src_mask'], batch['tar_mask'], src_embedding, tar_embedding) * combined_mask

        gamma = gamma_0.clone()
        src_coordinates = batch['src_coordinates']
        all_R, all_t = [], []
        not_converged = True
        iter_num = 0
        while(iter_num < self._max_iter):
            R_gamma, t_gamma, rmsd_gamma, rmsd_per_corr_gamma = weighted_kabsch_torch(src_coordinates,  batch['tar_coordinates'], gamma.float(),  batch['src_mask'],  batch['tar_mask'])
            all_R.append(R_gamma)
            all_t.append(t_gamma)
            # rotation = Rotation.from_matrix(R_gamma)
            # euler_angles = rotation.as_euler('xyz', degrees=False)
            # diff = np.mean(np.abs(euler_angles))
            # print(diff)
            # if diff < self._min_diff or iter_num >= self._max_iter:
                # not_converged = False
            src_coordinates = (torch.matmul(src_coordinates, R_gamma) + t_gamma.unsqueeze(1)) * batch['src_mask'].unsqueeze(-1).expand_as(batch['src_coordinates'])
            
            src_tgt_euc_dist = cdist_torch(batch['tar_coordinates'], src_coordinates, 3)
            gamma = (gamma_0 / (( 1 + (src_tgt_euc_dist / self.get_d0(batch['max_length'])[:, None, None])**2)**2))* combined_mask
            iter_num +=1
        
        R_total, t_total = self._compose_transformations(batch_size, all_R, all_t)
        
        loss: dict[str, torch.Tensor] = self._loss(R_total, t_total, batch['gt_R'], batch['gt_t'])
        
        return {'loss': loss['loss'], 'loss_dict': loss, 'pred_R': R_total, 'pred_t': t_total}
     
    def _mask_and_normalize_matrix(self, distance_matrix: torch.Tensor, src_mask: torch.Tensor, tar_mask: torch.Tensor, src_embedding: torch.Tensor, tar_embedding) -> list[tuple]:
        batch_size = distance_matrix.shape[0]
        t = torch.tensor([guess_best_alpha_torch(src_embedding[i,:][src_mask[i]], dim_num=tar_embedding.shape[-1], transpose=False) for i in range(batch_size)], device=self.device)
        R = torch.stack([softargmin_rows_torch(distance_matrix[i], t[i]) for i in range(batch_size)], dim=0)
        t = torch.tensor([guess_best_alpha_torch(tar_embedding[i,:][tar_mask[i]], dim_num=tar_embedding.shape[-1], transpose=False) for i in range(batch_size)] , device=self.device)
        C = torch.stack([softargmin_rows_torch(torch.transpose(distance_matrix, dim0=1, dim1=2)[i], t[i]) for i in range(batch_size)], dim=0)
        C = torch.transpose(C, dim0=1, dim1=2)
        B = torch.mul(R, C)
      
        return B
    
    def validation_step(self, batch, batch_idx):
        batch = move_batch_to_device(batch, self.device)
        batch_size = batch['tar_embedding'].shape[0]
        tar_embedding, src_embedding  = self._inout_tar_block(batch['tar_embedding'], batch['tar_coordinates'] ), self._inout_src_block(batch['src_embedding'], batch['src_coordinates'])
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
            # print(diff)
            if diff < self._min_diff or iter_num >= self._max_iter:
                not_converged = False
            src_coordinates = (torch.matmul(src_coordinates, R_gamma) + t_gamma.unsqueeze(1)) * batch['src_mask'].unsqueeze(-1).expand_as(batch['src_coordinates'])
            
            src_tgt_euc_dist = cdist_torch(batch['tar_coordinates'], src_coordinates, 3)
            gamma = (gamma_0 / (( 1 + (src_tgt_euc_dist / self.get_d0(batch['max_length'])[:, None, None])**2)**2))* combined_mask
            iter_num +=1        
           
        R_total, t_total = self._compose_transformations(batch_size, all_R, all_t)
        loss: dict[str, torch.Tensor] = self._loss(R_total, t_total, batch['gt_R'], batch['gt_t'])
        return {'loss': loss['loss'], 'loss_dict': loss, 'pred_R': R_total, 'pred_t': t_total}

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
    log_dir = os.path.join("logs", "learnable_softbbs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, start_time + ".log"), level=logging.INFO, format='%(message)s')

    logger = logging.getLogger(__name__)
    train = True
    base_data_path = os.path.join('/home/iscb/wolfson/hagairavid/ligand_alligner/ligands')
    data_path = 'results/baseline_results/2024-07-18_10-26-26_10000.csv'
    
    if train:
        train_dataset = ScannetDataset(data_path, base_data_path, 1500, min_cath = 3)
        train_loader  = DataLoader(train_dataset, batch_size=4, collate_fn=custom_collate_fn, num_workers=20)
        valid_dataset = ScannetDataset(data_path, base_data_path, 100, min_cath = 3)
        val_loader  = DataLoader(valid_dataset, batch_size=4, collate_fn=custom_collate_fn, num_workers=20)
        model = SoftBB()
        logger = pl.loggers.TensorBoardLogger('tb_logs/')

        trainer = L.Trainer(logger=logger, max_epochs=10, log_every_n_steps=10, accelerator= 'gpu' if torch.cuda.is_available() else 'cpu', profiler="simple")
        trainer.fit(model, train_loader, val_dataloaders=val_loader)
        