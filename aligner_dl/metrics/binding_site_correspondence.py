import logging
from torch.nn import Module

logger = logging.getLogger(__name__)


class BindingSiteCorrespondence(Module):
    """Weighted fraction of correspondences that fall within the binding site on both sides.

        An orthogonal, mapping-independent counterpart to the (symmetry-sensitive) ligand RMSD:
        for each correspondence i with soft-alignment weight w_i, checks whether both the source
        and target keypoints lie within the binding site (closest ligand atom within 4A).

        Expects:
            - batch contains `metadata` (per-sample dict with 'cath_degree').
            - outputs contains `corr_values` ([B, N]) and `corr_pocket_mask` ([B, N, 2], where
              [..., 0] is the source pocket mask and [..., 1] is the target pocket mask).
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.weighted_pocket_fraction_per_degree = {deg: 0.0 for deg in range(0, 9)}
        self.weighted_pocket_fraction_per_degree_protein = {deg: [] for deg in range(0, 9)}
        self.sample_metrics = {'pocket_fraction_per_sample': [], 'cath_degree_per_sample': []}
        self.count_per_degree = {deg: 0 for deg in range(0, 9)}
        self.total_count = 0

    def update(self, batch, outputs):
        batch_size = len(batch['metadata'])
        corr_values = outputs['corr_values']
        corr_pocket_mask = outputs['corr_pocket_mask']

        for batch_id in range(batch_size):
            metadata = batch['metadata'][batch_id]
            cath_degree = metadata['cath_degree']

            weights = corr_values[batch_id].float()
            both_in_pocket = (corr_pocket_mask[batch_id, :, 0] & corr_pocket_mask[batch_id, :, 1]).float()

            pocket_fraction = (weights * both_in_pocket).sum() / weights.sum()
            pocket_fraction = pocket_fraction.item()

            self.weighted_pocket_fraction_per_degree[cath_degree] += pocket_fraction
            self.weighted_pocket_fraction_per_degree_protein[cath_degree].append(pocket_fraction)
            self.count_per_degree[cath_degree] += 1
            self.sample_metrics['pocket_fraction_per_sample'].append(pocket_fraction)
            self.sample_metrics['cath_degree_per_sample'].append(cath_degree)

            self.total_count += 1

    def compute(self):
        weighted_pocket_fraction_per_degree_avg = {
            deg: (self.weighted_pocket_fraction_per_degree[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else 0.0
            for deg in range(0, 9)
        }

        overall = 0.0
        if self.total_count > 0:
            overall = sum(self.sample_metrics['pocket_fraction_per_sample']) / self.total_count

        return {
            'weighted_pocket_fraction_per_degree': weighted_pocket_fraction_per_degree_avg,
            'weighted_pocket_fraction_overall': overall,
            'weighted_pocket_fraction_per_degree_protein': self.weighted_pocket_fraction_per_degree_protein,
            **self.sample_metrics,
            'counts_per_degree': self.count_per_degree,
            'total_count': self.total_count,
        }
