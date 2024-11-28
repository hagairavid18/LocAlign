import logging
import torch
from torch.nn import Module

from models.utils.math import compute_rmsd_torch


logger = logging.getLogger(__name__)

class PocketRMSD(Module):
    def __init__(self):
        super(PocketRMSD, self).__init__()
        self.reset()

    def reset(self):
        # Initialize metrics for each degree 1 through 8
        self.pocket_rmsd_per_degree = {deg: 0 for deg in range(1, 9)}
        self.pocket_rmsd_first_iter_per_degree = {deg: 0 for deg in range(1, 9)}
        self.src_pocket_embeddings_scalar = {deg: 0 for deg in range(1, 9)}
        self.src_non_pocket_embeddings_scalar = {deg: 0 for deg in range(1, 9)}
        self.tar_pocket_embeddings_scalar = {deg: 0 for deg in range(1, 9)}
        self.tar_non_pocket_embeddings_scalar = {deg: 0 for deg in range(1, 9)}
        self.count_per_degree = {deg: 0 for deg in range(1, 9)}
        self.total_count = 0
        self.sample_metrics = {'cath_degree_per_sample': [], 'pocket_rmsd_per_sample': []}

    def update(self, batch, outputs):
        batch_size = len(batch['metadata'])
        for batch_id in range(batch_size):
            metadata = batch['metadata'][batch_id]
            cath_degree = metadata['cath_degree']

            # Compute transformations
            gt_T, pred_T = torch.eye(4), torch.eye(4)
            gt_T[:3, :3] = batch['gt_R'][batch_id]
            gt_T[:, 3] = batch['gt_t'][batch_id]
            pred_T[:3, :3] = outputs['pred_R'][batch_id]
            pred_T[:3, 3] = outputs['pred_t'][batch_id]

            # Compute RMSDs
            pocket_rmsd = compute_rmsd_torch(batch['src_pocket'], batch['gt_R'], batch['gt_t'], outputs['pred_R'], outputs['pred_t'], batch['src_pocket_mask'])[batch_id]

            # Update metrics
            self.pocket_rmsd_per_degree[cath_degree] += pocket_rmsd
            self.count_per_degree[cath_degree] += 1

            if "src_pocket_scalar_mean" in outputs['embeddings_dict']:
                self.src_pocket_embeddings_scalar[cath_degree]+= outputs['embeddings_dict']['src_pocket_scalar_mean']
                self.src_non_pocket_embeddings_scalar[cath_degree]+= outputs['embeddings_dict']['src_non_pocket_scalar_mean']
                self.tar_pocket_embeddings_scalar[cath_degree]+= outputs['embeddings_dict']['tar_pocket_scalar_mean']
                self.tar_non_pocket_embeddings_scalar[cath_degree]+= outputs['embeddings_dict']['tar_non_pocket_scalar_mean']

            self.sample_metrics['cath_degree_per_sample'].append(cath_degree)
            self.sample_metrics['pocket_rmsd_per_sample'].append(pocket_rmsd.cpu())

            # First iteration metrics
            if 'pred_first_R' in outputs:
                pred_T[:3, :3] = outputs['pred_first_R'][batch_id]
                pred_T[:3, 3] = outputs['pred_first_t'][batch_id]
                pocket_rmsd_first = compute_rmsd_torch(batch['src_pocket'], batch['gt_R'], batch['gt_t'], outputs['pred_first_R'], outputs['pred_first_t'], batch['src_pocket_mask'])[batch_id]
                self.pocket_rmsd_first_iter_per_degree[cath_degree] += pocket_rmsd_first
                
        self.total_count += batch_size

    def compute(self):
        # Calculate overall averages by summing all per-degree values and dividing by total count
        total_metrics = {
            'pocket_rmsd': sum(self.pocket_rmsd_per_degree.values()) / self.total_count if self.total_count > 0 else 0,
            'pocket_rmsd_iter0': sum(self.pocket_rmsd_first_iter_per_degree.values()) / self.total_count if self.total_count > 0 else 0,
        }

        # Calculate per-degree averages, handling zero counts
        per_degree_metrics = {
            'pocket_rmsd': {deg: (self.pocket_rmsd_per_degree[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else 0 for deg in range(1, 9)},
            'pocket_rmsd_iter0': {deg: (self.pocket_rmsd_first_iter_per_degree[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else 0 for deg in range(1, 9)},
        }
        
        pocket_embeddings_metrics = {
            'src_pocket_embeddings_scalar': {deg: (self.src_pocket_embeddings_scalar[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else 0 for deg in range(1, 9)},
            'tar_pocket_embeddings_scalar': {deg: (self.tar_pocket_embeddings_scalar[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else 0 for deg in range(1, 9)},
            'src_non_pocket_embeddings_scalar': {deg: (self.src_non_pocket_embeddings_scalar[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else 0 for deg in range(1, 9)},
            'tar_non_pocket_embeddings_scalar': {deg: (self.tar_non_pocket_embeddings_scalar[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else 0 for deg in range(1, 9)},
        }
        counts = {
            'counts_per_degree': self.count_per_degree,
            'total_count': self.total_count
        }

        # Return both total and per-degree metrics
        return {**total_metrics, **per_degree_metrics, **counts, **self.sample_metrics, **pocket_embeddings_metrics}