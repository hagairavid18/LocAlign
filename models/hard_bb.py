
from datetime import datetime
import os
import torch
from torch.utils.data import DataLoader
import lightning as L
from utils.kabsch import kabsch_torch
from models.utils.collate import custom_collate_fn


class HardBB(L.LightningModule):
    def __init__(self):
        super().__init__()
        self._validation_outputs = {}
       
    def validation_step(self, batch: dict[torch.Tensor], batch_idx: int):
        tar_embedding, src_embedding  = batch['tar_embedding'], batch['src_embedding']
        bsize = tar_embedding.shape[0]
        ref_expanded = tar_embedding.unsqueeze(2)  # Shape: (BATCH, A, 1, C)
        mov_expanded = src_embedding.unsqueeze(1)  # Shape: (BATCH, 1, B, C)

        # Calculate squared differences
        l2_distances = torch.sqrt(torch.sum((ref_expanded - mov_expanded) ** 2, dim=-1))  # Shape: (BATCH, A, B)
        min_indices_ref_to_mov = torch.argmin(l2_distances, axis=-1)
        min_indices_mov_to_ref = torch.argmin(l2_distances, axis=-2)
        # Find reciprocal pairs
        reciprocal_pairs = []
        for batch_id in range(bsize):
            for a_index in range(tar_embedding.shape[1]):
                b_index = min_indices_ref_to_mov[batch_id, a_index].item()
                if min_indices_mov_to_ref[batch_id, b_index] == a_index:
                    reciprocal_pairs.append((batch_id, a_index, b_index, l2_distances[batch_id, a_index, b_index]))
        curr_num_pairs = len(reciprocal_pairs) + 1
        
        for batch_id in range(bsize):
            while curr_num_pairs > len(reciprocal_pairs):
                curr_num_pairs = len(reciprocal_pairs)
                if curr_num_pairs <= 10:
                    break
                corresponded_tar_coord = batch['tar_coordinates'][batch_id, [pair[1] for pair in reciprocal_pairs]]
                corresponded_src_coord = batch['src_coordinates'][batch_id, [pair[2] for pair in reciprocal_pairs]]
                weights = [pair[3] for pair in reciprocal_pairs]
                R, t, rmsd, rmsd_per_corr = kabsch_torch(corresponded_src_coord, corresponded_tar_coord)
                print(rmsd)
                indices = torch.where(rmsd_per_corr <= 12)[0].tolist()
                reciprocal_pairs = [reciprocal_pairs[i] for i in indices]
            
            # print(torch.abs(R - torch.Tensor(batch['metadata'][batch_id]['rotations'][0][0])).mean())
            # if len(ransac_R) > 0:
            #     print(torch.abs(torch.Tensor(ransac_R[0]) - torch.Tensor(batch['metadata'][batch_id]['rotations'][0][0])).mean())
            # print(torch.abs(torch.Tensor(batch['metadata'][batch_id]['TMAligner_rotations']) - torch.Tensor(batch['metadata'][batch_id]['rotations'][0][0])).mean())
            # if batch['metadata'][batch_id]['DaliAligner_rotations'].any():
            #     print(torch.abs(torch.Tensor(batch['metadata'][batch_id]['DaliAligner_rotations']) - torch.Tensor(batch['metadata'][batch_id]['rotations'][0][0])).mean())
            # print()
        self._validation_outputs[batch['metadata'][0]['index']] = {
        'svd_R': R,
        'svd_t': t,
        }
     
    
    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

if __name__ == "__main__":
    from datasets import ScannetDataset
    from utils.misc import save_results_to_csv
    import logging
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", "bbs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, start_time + ".log"), level=logging.INFO, format='%(message)s')

    logger = logging.getLogger(__name__)

    data_path = '/home/iscb/wolfson/hagairavid/ligand_alligner/results/baseline_results/2024-07-11_10-55-38_57.csv'
    # data_path = '/home/iscb/wolfson/hagairavid/ligand_alligner/results/baseline_results/2024-07-17_16-01-08_3000.csv'
    # data_path = 'results/soft_bbs_results/2024-08-14_09-48-38_500.csv'
    data_path = 'results/soft_bbs_results/2024-08-14_22-22-35_2000.csv'
    base_data_path = os.path.join('/home/iscb/wolfson/hagairavid/ligands')
    dataset = ScannetDataset(data_path, base_data_path, 2000)
    val_loader  = DataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn, num_workers=20)
    model = HardBB()

    trainer = L.Trainer()
    trainer.validate(model, dataloaders=val_loader)
    output_df = dataset._df.copy()

    output_df['HardBBS_rotations'] = [[] for _ in range(len(output_df))]
    output_df['HardBBS_translations'] = [[] for _ in range(len(output_df))]
    for row_idx, values in model._validation_outputs.items():
        output_df.at[row_idx,'HardBBS_translations'] = values['svd_t'].tolist()
        output_df.at[row_idx,'HardBBS_rotations'] = values['svd_R'].tolist()
    
    save_results_to_csv(output_df, start_time, base_dir="hard_bbs_results")