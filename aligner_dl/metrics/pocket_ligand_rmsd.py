import logging
from torch.nn import Module

from losses.soft_bb_loss import LigandLoss


logger = logging.getLogger(__name__)


class PocketRMSD(Module):
    def __init__(self):
        super(PocketRMSD, self).__init__()
        self.reset()

    def reset(self):
        # Initialize metrics for each degree 1 through 8
        self.ligand_rmsd_per_degree = {deg: 0 for deg in range(0, 9)}
        self.corr_rmsd_per_degree = {deg: 0 for deg in range(0, 9)}
        self.embedding_similarity_per_degree = {deg: 0 for deg in range(0, 9)}
        self.gap_per_degree = {deg: 0 for deg in range(0, 9)}
        self.count_per_degree = {deg: 0 for deg in range(0, 9)}
        self.total_count = 0
        self.sample_metrics = {'cath_degree_per_sample': [], "ligand_rmsd_per_sample" : [], 'pair_infos': []}

        # Initialize a dictionary to store protein names and their pocket_rmsd per degree
        self.pair_infos_per_degree = {deg: [] for deg in range(0, 9)}
        self.ligand_rmsd_per_degree_protein = {deg: [] for deg in range(0, 9)}
        self.corr_rmsd_per_degree_protein = {deg: [] for deg in range(0, 9)}
        self.gap_per_degree_protein = {deg: [] for deg in range(0, 9)}
        self.embedding_similarity_per_degree_protein = {deg: [] for deg in range(0, 9)}
        self.radius_of_gyration_per_degree_protein = {deg: [] for deg in range(0, 9)}
        self._ligand_rmsd_metric = LigandLoss(return_non_linear=False)

    def update(self, batch, outputs):
        batch_size = len(batch['metadata'])
        for batch_id in range(batch_size):
            metadata = batch['metadata'][batch_id]
            cath_degree = metadata['cath_degree']
            pair_info = batch['metadata'][batch_id].copy()  # Make a copy to avoid modifying the original
            for key in ['rotations', 'translations', 'rmse', 'coverage']:
                pair_info.pop(key)

            # Compute RMSDs
            ligand_rmsd = self._ligand_rmsd_metric(batch, outputs['transformation_dict']['pred_R'], outputs['transformation_dict']['pred_t'], reduce=False)[batch_id]

            # Update metrics
            self.ligand_rmsd_per_degree[cath_degree] += ligand_rmsd
            self.count_per_degree[cath_degree] += 1

            self.sample_metrics['cath_degree_per_sample'].append(cath_degree)
            self.sample_metrics['ligand_rmsd_per_sample'].append(ligand_rmsd.cpu())
            
            self.sample_metrics['pair_infos'].append(pair_info)  # Store the protein name

            # Update the dictionary with protein names and pocket_rmsd per degree
            self.pair_infos_per_degree[cath_degree].append(pair_info)
            self.ligand_rmsd_per_degree_protein[cath_degree].append(ligand_rmsd.cpu())
            self.embedding_similarity_per_degree_protein[cath_degree].append(outputs['loss_dict']['per_sample']['embedding'][batch_id].cpu())
            self.corr_rmsd_per_degree_protein[cath_degree].append(outputs['loss_dict']['per_sample']['corr_rmsd'][batch_id].cpu())
            self.gap_per_degree_protein[cath_degree].append(outputs['loss_dict']['per_sample']['gap'][batch_id].cpu())
            self.radius_of_gyration_per_degree_protein[cath_degree].append(outputs['loss_dict']['per_sample']['radius'][batch_id].cpu())

        self.total_count += batch_size

    def compute(self):
        # Calculate overall averages by summing all per-degree values and dividing by total count
        total_metrics = {}

        # Calculate per-degree averages, handling zero counts
        per_degree_metrics = {
            'ligand_rmsd': {deg: (self.ligand_rmsd_per_degree[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else 0 for deg in range(0, 9)},
            # 'corr_rmsd': {deg: (self.corr_rmsd_per_degree[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else 0 for deg in range(0, 9)},
            # 'gap': {deg: (self.gap_per_degree[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else 0 for deg in range(0, 9)},
            # 'embedding_similarity': {deg: (self.embedding_similarity_per_degree[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else 0 for deg in range(0, 9)},
        }
        
        counts = {
            'counts_per_degree': self.count_per_degree,
            'total_count': self.total_count
        }

        # same for ligand rmsd if needed
        ligand_rmsd_below_4_per_degree = {deg: 0 for deg in range(0, 9)}
        total_ligand_rmsd_below_4 = 0
        for rmsd, degree in zip(self.sample_metrics['ligand_rmsd_per_sample'], self.sample_metrics['cath_degree_per_sample']):
            if rmsd < 4:
                ligand_rmsd_below_4_per_degree[degree] += 1
                total_ligand_rmsd_below_4 += 1

        proportion_below_4_per_degree_ligand = {
            deg: (ligand_rmsd_below_4_per_degree[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else 0 for deg in range(0, 9)
        }
        return {   # Calculate proportion of RMSD < 4 per degree and overall
            **total_metrics, **per_degree_metrics, **counts, **self.sample_metrics, 
            'pair_infos_per_degree': self.pair_infos_per_degree, 
            'ligand_rmsd_per_degree_protein': self.ligand_rmsd_per_degree_protein,
            'corr_rmsd_per_degree_protein': self.corr_rmsd_per_degree_protein,
            'gap_per_degree_protein': self.gap_per_degree_protein,
            'radius_per_degree_protein': self.radius_of_gyration_per_degree_protein,
            'embedding_similarity_per_degree_protein': self.embedding_similarity_per_degree_protein,
            'ligand_rmsd_below_4_proportion_per_degree': proportion_below_4_per_degree_ligand,
            'ligand_rmsd_below_4_total_proportion': total_ligand_rmsd_below_4
        }
