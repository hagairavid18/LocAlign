import logging
from torch.nn import Module

from models.utils.math import compute_rmsd_torch


logger = logging.getLogger(__name__)


class PocketRMSD(Module):
    def __init__(self):
        super(PocketRMSD, self).__init__()
        self.reset()

    def reset(self):
        # Initialize metrics for each degree 1 through 8
        self.pocket_rmsd_per_degree = {deg: 0 for deg in range(0, 9)}
        self.ligand_rmsd_per_degree = {deg: 0 for deg in range(0, 9)}
        self.corr_rmsd_per_degree = {deg: 0 for deg in range(0, 9)}
        self.count_per_degree = {deg: 0 for deg in range(0, 9)}
        self.total_count = 0
        self.sample_metrics = {'cath_degree_per_sample': [], 'pocket_rmsd_per_sample': [], "ligand_rmsd_per_sample" : [], "corr_rmsd_per_sample": [], 'pair_infos': []}

        # Initialize a dictionary to store protein names and their pocket_rmsd per degree
        self.pair_infos_per_degree = {deg: [] for deg in range(0, 9)}
        self.pocket_rmsd_per_degree_protein = {deg: [] for deg in range(0, 9)}
        self.corr_rmsd_per_degree_protein = {deg: [] for deg in range(0, 9)}

    def update(self, batch, outputs):
        batch_size = len(batch['metadata'])
        for batch_id in range(batch_size):
            metadata = batch['metadata'][batch_id]
            cath_degree = metadata['cath_degree']
            pair_info = batch['metadata'][batch_id].copy()  # Make a copy to avoid modifying the original
            for key in ['rotations', 'translations', 'rmse', 'coverage']:
                pair_info.pop(key)
            # pair_info.pop([['rotations', 'translations', 'rmse', 'coverage']])

            # Compute RMSDs
            pocket_coordinates = batch['src_pocket_frames'][:, :, 0, :]
            pocket_rmsd = compute_rmsd_torch(pocket_coordinates, batch['gt_R'], batch['gt_t'], outputs['transformation_dict']['pred_R'], outputs['transformation_dict']['pred_t'], batch['src_pocket_mask'])[batch_id]
            ligand_rmsd = compute_rmsd_torch(batch['src_ligand_coordinates'], batch['gt_R'], batch['gt_t'], outputs['transformation_dict']['pred_R'], outputs['transformation_dict']['pred_t'], batch['src_ligand_mask'])[batch_id]
            pocket_rmsd = compute_rmsd_torch(pocket_coordinates, batch['gt_R'], batch['gt_t'], outputs['transformation_dict']['pred_R'], outputs['transformation_dict']['pred_t'], batch['src_pocket_mask'])[batch_id]
            ligand_rmsd = compute_rmsd_torch(batch['src_ligand_coordinates'], batch['gt_R'], batch['gt_t'], outputs['transformation_dict']['pred_R'], outputs['transformation_dict']['pred_t'], batch['src_ligand_mask'])[batch_id]

            # Update metrics
            self.pocket_rmsd_per_degree[cath_degree] += pocket_rmsd
            self.ligand_rmsd_per_degree[cath_degree] += ligand_rmsd
            self.count_per_degree[cath_degree] += 1

            self.sample_metrics['cath_degree_per_sample'].append(cath_degree)
            self.sample_metrics['pocket_rmsd_per_sample'].append(pocket_rmsd.cpu())
            self.sample_metrics['ligand_rmsd_per_sample'].append(ligand_rmsd.cpu())
            
            self.sample_metrics['pair_infos'].append(pair_info)  # Store the protein name

            # Update the dictionary with protein names and pocket_rmsd per degree
            self.pair_infos_per_degree[cath_degree].append(pair_info)
            self.pocket_rmsd_per_degree_protein[cath_degree].append(pocket_rmsd.cpu())

            # update corr_rmsd per degree
            if 'corr_rmsd' in outputs['transformation_dict']:
                corr_rmsd = outputs['transformation_dict']['corr_rmsd'][batch_id].cpu()
                self.sample_metrics['corr_rmsd_per_sample'].append(corr_rmsd)
                self.corr_rmsd_per_degree[cath_degree] += corr_rmsd
                self.corr_rmsd_per_degree_protein[cath_degree].append(corr_rmsd)

        self.total_count += batch_size

    def compute(self):
        # Calculate overall averages by summing all per-degree values and dividing by total count
        total_metrics = {
            'pocket_rmsd': sum(self.pocket_rmsd_per_degree.values()) / self.total_count if self.total_count > 0 else 0        }

        # Calculate per-degree averages, handling zero counts
        per_degree_metrics = {
            'pocket_rmsd': {deg: (self.pocket_rmsd_per_degree[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else 0 for deg in range(0, 9)},
            'ligand_rmsd': {deg: (self.ligand_rmsd_per_degree[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else 0 for deg in range(0, 9)},
            'corr_rmsd': {deg: (self.corr_rmsd_per_degree[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else 0 for deg in range(0, 9)},
        }
        
        counts = {
            'counts_per_degree': self.count_per_degree,
            'total_count': self.total_count
        }

        # Return both total and per-degree metrics, including protein names and pocket RMSD for each degree
        pocket_rmsd_below_4_per_degree = {deg: 0 for deg in range(0, 9)}
        total_pocket_rmsd_below_4 = 0
        for rmsd, degree in zip(self.sample_metrics['pocket_rmsd_per_sample'], self.sample_metrics['cath_degree_per_sample']):
            if rmsd < 4:
                pocket_rmsd_below_4_per_degree[degree] += 1
                total_pocket_rmsd_below_4 += 1

        # same for ligand rmsd if needed
        ligand_rmsd_below_2_per_degree = {deg: 0 for deg in range(0, 9)}
        total_ligand_rmsd_below_2 = 0
        for rmsd, degree in zip(self.sample_metrics['ligand_rmsd_per_sample'], self.sample_metrics['cath_degree_per_sample']):
            if rmsd < 2:
                ligand_rmsd_below_2_per_degree[degree] += 1
                total_ligand_rmsd_below_2 += 1



        proportion_below_4_per_degree_pocket = {
            deg: (pocket_rmsd_below_4_per_degree[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else 0 for deg in range(0, 9)
        }
        proportion_below_2_per_degree_ligand = {
            deg: (ligand_rmsd_below_2_per_degree[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else 0 for deg in range(0, 9)
        }

        total_proportion_below_4_pocket = total_pocket_rmsd_below_4 / self.total_count if self.total_count > 0 else 0
        return {   # Calculate proportion of RMSD < 4 per degree and overall
            **total_metrics, **per_degree_metrics, **counts, **self.sample_metrics, 
            'pair_infos_per_degree': self.pair_infos_per_degree, 
            'pocket_rmsd_per_degree_protein': self.pocket_rmsd_per_degree_protein,
            'corr_rmsd_per_degree_protein': self.corr_rmsd_per_degree_protein,
            'pocket_rmsd_below_4_proportion_per_degree': proportion_below_4_per_degree_pocket,
            'pocket_rmsd_below_4_total_proportion': total_proportion_below_4_pocket,
            'ligand_rmsd_below_2_proportion_per_degree': proportion_below_2_per_degree_ligand,
            'ligand_rmsd_below_2_total_proportion': total_ligand_rmsd_below_2
        }
