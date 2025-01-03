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
        self.ligand_rmsd_per_degree = {deg: 0 for deg in range(1, 9)}
        self.ligand_rmsd_first_iter_per_degree = {deg: 0 for deg in range(1, 9)}
        self.src_pocket_embeddings_scalar = {deg: 0 for deg in range(1, 9)}
        self.src_non_pocket_embeddings_scalar = {deg: 0 for deg in range(1, 9)}
        self.tar_pocket_embeddings_scalar = {deg: 0 for deg in range(1, 9)}
        self.tar_non_pocket_embeddings_scalar = {deg: 0 for deg in range(1, 9)}
        self.count_per_degree = {deg: 0 for deg in range(1, 9)}
        self.total_count = 0
        self.sample_metrics = {'cath_degree_per_sample': [], 'pocket_rmsd_per_sample': [], 'pair_infos': []}

        # Initialize a dictionary to store protein names and their pocket_rmsd per degree
        self.pair_infos_per_degree = {deg: [] for deg in range(1, 9)}
        self.pocket_rmsd_per_degree_protein = {deg: [] for deg in range(1, 9)}

    def update(self, batch, outputs):
        batch_size = len(batch['metadata'])
        for batch_id in range(batch_size):
            metadata = batch['metadata'][batch_id]
            cath_degree = metadata['cath_degree']
            pair_info = (batch['metadata'][0]['Ligand_ID'], batch['metadata'][0]['mov_protein'], batch['metadata'][0]['ref_protein'], batch['metadata'][0]['bbr'][0][0])

            # Compute transformations
            gt_T, pred_T = torch.eye(4), torch.eye(4)
            gt_T[:3, :3] = batch['gt_R'][batch_id]
            gt_T[:, 3] = batch['gt_t'][batch_id]
            pred_T[:3, :3] = outputs['transformation_dict']['pred_R'][batch_id]
            pred_T[:3, 3] = outputs['transformation_dict']['pred_t'][batch_id]

            # Compute RMSDs
            pocket_rmsd = compute_rmsd_torch(batch['src_pocket'], batch['gt_R'], batch['gt_t'], outputs['transformation_dict']['pred_R'], outputs['transformation_dict']['pred_t'], batch['src_pocket_mask'])[batch_id]
            ligand_rmsd = compute_rmsd_torch(batch['src_ligand_coordinates'], batch['gt_R'], batch['gt_t'], outputs['transformation_dict']['pred_R'], outputs['transformation_dict']['pred_t'], batch['src_ligand_mask'])[batch_id]

            # Update metrics
            self.pocket_rmsd_per_degree[cath_degree] += pocket_rmsd
            self.ligand_rmsd_per_degree[cath_degree] += ligand_rmsd
            self.count_per_degree[cath_degree] += 1

            if 'embeddings_dict' in outputs and "src_pocket_scalar_mean" in outputs['embeddings_dict']:
                self.src_pocket_embeddings_scalar[cath_degree] += outputs['embeddings_dict']['src_pocket_scalar_mean']
                self.src_non_pocket_embeddings_scalar[cath_degree] += outputs['embeddings_dict']['src_non_pocket_scalar_mean']
                self.tar_pocket_embeddings_scalar[cath_degree] += outputs['embeddings_dict']['tar_pocket_scalar_mean']
                self.tar_non_pocket_embeddings_scalar[cath_degree] += outputs['embeddings_dict']['tar_non_pocket_scalar_mean']

            self.sample_metrics['cath_degree_per_sample'].append(cath_degree)
            self.sample_metrics['pocket_rmsd_per_sample'].append(pocket_rmsd.cpu())
            self.sample_metrics['pair_infos'].append(pair_info)  # Store the protein name

            # Update the dictionary with protein names and pocket_rmsd per degree
            self.pair_infos_per_degree[cath_degree].append(pair_info)
            self.pocket_rmsd_per_degree_protein[cath_degree].append(pocket_rmsd.cpu())

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
            'ligand_rmsd': {deg: (self.ligand_rmsd_per_degree[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else 0 for deg in range(1, 9)},
            'ligand_rmsd_iter0': {deg: (self.ligand_rmsd_first_iter_per_degree[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else 0 for deg in range(1, 9)},
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

        # Return both total and per-degree metrics, including protein names and pocket RMSD for each degree
        rmsd_below_4_per_degree = {deg: 0 for deg in range(1, 9)}
        total_rmsd_below_4 = 0
        for rmsd, degree in zip(self.sample_metrics['pocket_rmsd_per_sample'], self.sample_metrics['cath_degree_per_sample']):
            if rmsd < 4:
                rmsd_below_4_per_degree[degree] += 1
                total_rmsd_below_4 += 1

        proportion_below_4_per_degree = {
            deg: (rmsd_below_4_per_degree[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else 0 for deg in range(1, 9)
        }
        total_proportion_below_4 = total_rmsd_below_4 / self.total_count if self.total_count > 0 else 0
        return {   # Calculate proportion of RMSD < 4 per degree and overall
            **total_metrics, **per_degree_metrics, **counts, **self.sample_metrics, 
            **pocket_embeddings_metrics, 
            'pair_infos_per_degree': self.pair_infos_per_degree, 
            'pocket_rmsd_per_degree_protein': self.pocket_rmsd_per_degree_protein,
             'rmsd_below_4_proportion_per_degree': proportion_below_4_per_degree,
            'rmsd_below_4_total_proportion': total_proportion_below_4,
        }
