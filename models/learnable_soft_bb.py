
from datetime import datetime
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L
import pytorch_lightning as pl

from utils.kabsch import weighted_kabsch_torch
from models.utils.collate import custom_collate_fn
from utils.deepbbs_utils import *
from scipy.spatial.transform import Rotation
torch.autograd.set_detect_anomaly(True)
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


def plot_transformed_point_clouds(src_coordinates, tar_coordinates, Rs, ts, batch_idx=0, postfix = "", compose: bool = True):
    # Select the batch
    src = src_coordinates[batch_idx]
    tar = tar_coordinates[batch_idx]
    
    # Define perspectives
    perspectives = [(30, 45), (60, 90), (90, 0)]
    
    # Initialize the transformed coordinates with the original
    src_transformed = src.clone()
    tar_transformed = tar.clone()
    
    fig = plt.figure(figsize=(15, 15))
    
    for idx, (R_gamma, t_gamma) in enumerate(zip(Rs, ts)):
        # Ensure R_gamma and t_gamma are the correct shapes
         # Apply the transformation
        if compose:
            # src_transformed = (torch.matmul(src_transformed, R_gamma.transpose(1, 2)) + t_gamma.unsqueeze(0))[0]
            src_transformed = (torch.matmul(src_transformed, R_gamma) + t_gamma.unsqueeze(0))[0]
        else:
            # src_transformed = (torch.matmul(src.clone(), R_gamma.transpose(1, 2)) + t_gamma.unsqueeze(0))[0]
            src_transformed = (torch.matmul(src.clone(), R_gamma) + t_gamma.unsqueeze(0))[0]
        
        for i, (elev, azim) in enumerate(perspectives):
            ax = fig.add_subplot(len(Rs), len(perspectives), idx * len(perspectives) + i + 1, projection='3d')

            # Plot source coordinates
            ax.scatter(src_transformed[:, 0], src_transformed[:, 1], src_transformed[:, 2], c='r', marker='o', label='Source', s=1)

            # Plot target coordinates
            ax.scatter(tar_transformed[:, 0], tar_transformed[:, 1], tar_transformed[:, 2], c='b', marker='^', label='Target', s=1)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Transform {idx+1} - View {i+1}')
            
            # Set view perspective
            ax.view_init(elev=elev, azim=azim)
    
       
        # tar_transformed = (torch.matmul(tar_transformed, R_gamma) + t_gamma.unsqueeze(0))[0]
        
    plt.suptitle(f'Point Cloud Visualizations for Batch {batch_idx}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle
    plt.savefig(f'plots/{postfix}.png', dpi=500)
    plt.show()


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

class Loss(nn.Module):
    def __init__(self, translation_weight: float = 0.01):
        super(Loss, self).__init__()
        self._translation_weight = translation_weight
    
    def forward(self, rotation_ab_pred, translation_ab_pred, rotation_ab, translation_ab, batch_size):
        identity = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
    #    ind_mask = (cdist_torch(transform_point_cloud(src, rotation_ab, translation_ab), target, points_dim=3).min(dim=2).values < 0.05)
        rotation_mse = F.mse_loss(torch.matmul(rotation_ab_pred, rotation_ab), identity)
        translation_mse = F.mse_loss(translation_ab_pred, translation_ab[:, :3])
            #    + 0.95**epoch * ((src_corr - transform_point_cloud(src, rotation_ab, translation_ab)) ** 2).sum(dim=1).view(-1)[ind_mask.view(-1)].mean()
        return {'loss': rotation_mse + self._translation_weight * translation_mse, 'rot_loss': rotation_mse, 'tran_loss': translation_mse}



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
        self._inout_tar_block = FeatureCoordinateBlock(128 ,256, 256)
        self._inout_src_block = FeatureCoordinateBlock(128 ,256, 256)
        self._loss = Loss()
        self.validation_step_outputs = []
        self.alpha_factor = 4
        self.eps = 0.00001
        self._min_diff = 0.05
        self._max_iter = 10
    
    # def on_after_backward(self):
    #     """Logs gradients after backpropagation"""
    #     for name, param in self.named_parameters():
    #         if param.requires_grad and param.grad is not None:
    #             # Log gradients
    #             self.logger.experiment.add_histogram(f'{name}_gradients', param.grad, self.global_step, on_step=True)


    def training_step(self, batch: dict[torch.Tensor], batch_idx: int):
        print(batch['metadata'][0]['index'])
        tar_embedding, src_embedding  = self._inout_tar_block(batch['tar_embedding'], batch['tar_coordinates'] ), self._inout_src_block(batch['src_embedding'], batch['src_coordinates'])
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
        gamma = gamma_0.clone()
        src_coordinates = batch['src_coordinates']

        R_gamma, t_gamma, rmsd_gamma, rmsd_per_corr_gamma = weighted_kabsch_torch(src_coordinates,  batch['tar_coordinates'], gamma.float(),  batch['src_mask'],  batch['tar_mask'])
        src_coordinates = (torch.matmul(src_coordinates, R_gamma) + t_gamma.unsqueeze(1)) * batch['src_mask'].unsqueeze(-1).expand_as(batch['src_coordinates'])
        
        src_tgt_euc_dist = cdist_torch(batch['tar_coordinates'], src_coordinates, 3)
        gamma = (gamma_0 / (( 1 + (src_tgt_euc_dist / d_0[:, None, None])**2)**2))* combined_mask
           
        loss: dict[str, torch.Tensor] = self._loss(R_gamma, t_gamma, batch['gt_R'], batch['gt_t'], batch_size)
        # self.logger.experiment.add_scalar('myloss', loss['loss'], self.global_step)
        # Perform the backward pass manually if needed
        # loss['loss'].backward()

        # # Now log the gradients manually
        # for name, param in self.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         # Log the gradients
        #         self.logger.experiment.add_histogram(f'{name}_gradients', param.grad, self.global_step)
        
        self.log('train_loss', loss['loss'], batch_size=batch_size, prog_bar=True, on_step=True, on_epoch=True)
        # # Access the optimizer (make sure the optimizer is defined in configure_optimizers)
        # optimizer = self.optimizers()

        # # Manually perform optimizer step
        # optimizer.step()

        # # Manually zero the gradients
        # optimizer.zero_grad()
        return loss
     
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
            gamma = (gamma_0 / (( 1 + (src_tgt_euc_dist / d_0)**2)**2))* combined_mask
            iter_num +=1
           
        # plot_transformed_point_clouds(batch['src_coordinates'], batch['tar_coordinates'], all_R, all_t, postfix=f"pred_{batch['metadata'][0]['index']}")
        
           
        R_total = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)  # Initial rotation matrix (B x 3 x 3)
        t_total = torch.zeros(batch_size, 3)  # Initial translation vector (B x 3)

        # Compose the transformations by applying from the right (X R + t)
        for i, (R, t) in enumerate(zip(all_R, all_t)):
            # Multiply the composed rotation from the right
            R_total = R_total @ R
            
            # Accumulate translation, applying rotation to previous translations
            t_total = t_total @ R + t
                
        t_total = t_total.squeeze(1)
        # R_to_plot = [torch.Tensor(batch['metadata'][0]['rotations'][0]), torch.Tensor(batch['metadata'][0]['TMaligner_rotations']), all_R[0], R_total]
        # t_to_plot = [torch.Tensor(batch['metadata'][0]['translations'][0][:,:3]), torch.Tensor(batch['metadata'][0]['TMaligner_translations']), all_t[0], t_total]
        # plot_transformed_point_clouds(batch['src_coordinates'], batch['tar_coordinates'], R_to_plot, t_to_plot, postfix=batch['metadata'][0]['index'], iteration=iter_num, compose = False)

        # print(torch.Tensor(batch['metadata'][0]['rotations'][0][0]))
        # print(torch.Tensor(batch['metadata'][0]['TMaligner_rotations']))
        # print(all_R[0])
        # print(R_total)

        # print(torch.Tensor(batch['metadata'][0]['translations'][0][0]))
        # print(torch.Tensor(batch['metadata'][0]['TMaligner_translations']))
        # print(all_t[0])
        # print(t_total)
        # print(batch['metadata'][0]['index'])

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
    base_data_path = os.path.join('/home/iscb/wolfson/hagairavid/ligand_alligner/ligands')
    
    
    if train:
        torch.autograd.set_detect_anomaly(True)
        train_dataset = ScannetDataset(data_path, base_data_path, 1000)
        train_loader  = DataLoader(train_dataset, batch_size=4, collate_fn=custom_collate_fn, num_workers=10)
        model = SoftBB()
        logger = pl.loggers.TensorBoardLogger('tb_logs/')

        trainer = L.Trainer(logger=logger, max_epochs=50, log_every_n_steps=1)
        trainer.fit(model, train_loader)
    
    else:
        valid_dataset = ScannetDataset(data_path, base_data_path, 2000)
        val_loader  = DataLoader(valid_dataset, batch_size=4, collate_fn=custom_collate_fn, num_workers=30)
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