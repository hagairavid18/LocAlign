
from datetime import datetime
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L
from aligners import RANSACaligner


def custom_collate_fn(batch: list[dict[torch.Tensor, dict]]):
    # Assume batch is a list of dictionaries
    batch_dict = {}
    for key, val in batch[0].items():
        # Stack all tensors for a given key
        if isinstance(val, dict):
            batch_dict[key] = [item[key] for item in batch]
        elif isinstance(val, int):
            batch_dict[key] = [item[key] for item in batch]
        else:
            batch_dict[key] = torch.stack([item[key] for item in batch])
            
    return batch_dict

import torch


def kabsch_torch(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.
    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = torch.mean(P, dim=0)
    centroid_Q = torch.mean(Q, dim=0)

    # Optimal translation
    t = centroid_Q - centroid_P

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = torch.matmul(p.transpose(0, 1), q)

    # SVD
    U, S, Vt = torch.linalg.svd(H)

    # Validate right-handed coordinate system
    if torch.det(torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))) < 0.0:
        Vt[:, -1] *= -1.0

    # Optimal rotation
    R = torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))
    t = centroid_Q - torch.matmul(R, centroid_P)

    diff = (torch.mm(P, R.T) + t) - Q
    # diff = torch.matmul(p, R.transpose(0, 1)) - q
    rmsd_per_bb = torch.sqrt(torch.sum(torch.square(diff), axis=1))
    rmsd = torch.sqrt(torch.mean(torch.sum(torch.square(diff), axis=1)))

    return R.T, t, rmsd, rmsd_per_bb


class HardBB(L.LightningModule):
    def __init__(self):
        super().__init__()
        self._validation_outputs = {}
        self._aligner = RANSACaligner(criterion_threshold=5, cluster_thresh=100000, save_plots=False, n_ransac=100)
       
    def validation_step(self, batch: dict[torch.Tensor], batch_idx: int):
        ref_embedding, mov_embedding  = batch['ref_embedding'], batch['mov_embedding']
        bsize = ref_embedding.shape[0]
        ref_expanded = ref_embedding.unsqueeze(2)  # Shape: (BATCH, A, 1, C)
        mov_expanded = mov_embedding.unsqueeze(1)  # Shape: (BATCH, 1, B, C)

        # Calculate squared differences
        l2_distances = torch.sqrt(torch.sum((ref_expanded - mov_expanded) ** 2, dim=-1))  # Shape: (BATCH, A, B)
        min_indices_ref_to_mov = torch.argmin(l2_distances, axis=-1)
        min_indices_mov_to_ref = torch.argmin(l2_distances, axis=-2)
        # Find reciprocal pairs
        reciprocal_pairs = []
        for batch_id in range(bsize):
            for a_index in range(ref_embedding.shape[1]):
                b_index = min_indices_ref_to_mov[batch_id, a_index].item()
                if min_indices_mov_to_ref[batch_id, b_index] == a_index:
                    reciprocal_pairs.append((batch_id, a_index, b_index))
        curr_num_pairs = len(reciprocal_pairs) + 1
        
        for batch_id in range(bsize):
            iter_0 = True
            while curr_num_pairs > len(reciprocal_pairs):
                curr_num_pairs = len(reciprocal_pairs)
                if curr_num_pairs <= 10:
                    break
                corresponded_ref_coord = batch['ref_coordinates'][batch_id, [pair[1] for pair in reciprocal_pairs]]
                corresponded_mov_coord = batch['mov_coordinates'][batch_id, [pair[2] for pair in reciprocal_pairs]]
                if iter_0:
                    ransac_R, ransac_t, ransac_rmsd, ransac_coverage = self._aligner.impose_structure(corresponded_ref_coord, corresponded_mov_coord, save_dir="/home/iscb/wolfson/hagairavid/ligand_aligner")
                    iter_0 = False
                R, t, rmsd, rmsd_per_corr = kabsch_torch(corresponded_mov_coord, corresponded_ref_coord)
                print(rmsd)
                indices = torch.where(rmsd_per_corr <= 10)[0].tolist()
                reciprocal_pairs = [reciprocal_pairs[i] for i in indices]
            
            # print(torch.abs(R - torch.Tensor(batch['metadata'][batch_id]['rotations'][0][0])).mean())
            # if len(ransac_R) > 0:
            #     print(torch.abs(torch.Tensor(ransac_R[0]) - torch.Tensor(batch['metadata'][batch_id]['rotations'][0][0])).mean())
            # print(torch.abs(torch.Tensor(batch['metadata'][batch_id]['TMaligner_rotations']) - torch.Tensor(batch['metadata'][batch_id]['rotations'][0][0])).mean())
            # if batch['metadata'][batch_id]['DaliAligner_rotations'].any():
            #     print(torch.abs(torch.Tensor(batch['metadata'][batch_id]['DaliAligner_rotations']) - torch.Tensor(batch['metadata'][batch_id]['rotations'][0][0])).mean())
            # print()
        self._validation_outputs[batch['row_idx'][0]] = {
        'svd_R': R,
        'svd_t': t,
        'ransac_R': torch.Tensor(ransac_R[0]) if len(ransac_R) else torch.Tensor([]),
        'ransac_t': torch.Tensor(ransac_t[0][:3]) if len(ransac_t) else torch.Tensor([]),
        }
     
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

if __name__ == "__main__":
    from datasets import ScannetDataset
    from utils.misc import save_results_to_csv
    import logging
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(filename=os.path.join("logs", "hardbbs_" + start_time + ".log"), level=logging.INFO, format='%(message)s')

    logger = logging.getLogger(__name__)

    data_path = '/home/iscb/wolfson/hagairavid/ligand_aligner/results/baseline_results/2024-07-11_10-55-38_57.csv'
    data_path = '/home/iscb/wolfson/hagairavid/ligand_aligner/results/baseline_results/2024-07-17_16-01-08_3000.csv'
    base_data_path = os.path.join('/home/iscb/wolfson/hagairavid/ligand_aligner/ligands')
    dataset = ScannetDataset(data_path, base_data_path, 500)
    val_loader  = DataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn)
    model = HardBB()

    trainer = L.Trainer()
    trainer.validate(model, dataloaders=val_loader)
    output_df = dataset._df.copy()

    output_df['HardBBS_rotations'] = [[] for _ in range(len(output_df))]
    output_df['HardBBS_translations'] = [[] for _ in range(len(output_df))]
    output_df['HardBBSRansac_rotations'] = [[] for _ in range(len(output_df))]
    output_df['HardBBSRansac_translations'] = [[] for _ in range(len(output_df))]
    for row_idx, values in model._validation_outputs.items():
        output_df.at[row_idx,'HardBBS_translations'] = values['svd_t'].tolist()
        output_df.at[row_idx,'HardBBS_rotations'] = values['svd_R'].tolist()
        output_df.at[row_idx,'HardBBSRansac_translations'] = values['ransac_t'].tolist()
        output_df.at[row_idx,'HardBBSRansac_rotations'] = values['ransac_R'].tolist()
    
    save_results_to_csv(output_df, start_time, base_dir="hard_bbs_results")

    
    print('finished validation')

    # trainer.fit(model=autoencoder, train_dataloaders=train_loader)