import logging
from torch.nn import Module
import torch

logger = logging.getLogger(__name__)


class AtomTypeCorrespondence(Module):
    """Compute weighted fraction of correspondences preserving atom type.

        Expects:
            - batch contains `src_residue_indices` / `tar_residue_indices` (per-atom residue id numbers),
              `src_mask` / `tar_mask`, and the lists `src_residue_ids`, `tar_residue_ids`,
              plus `src_aa_to_atom_indices` and `tar_aa_to_atom_indices` (per-residue atom index lists).
            - outputs contains `corr_values` and `corr_indices` (residue-level pairs shaped [B, N, 2]).
        This implementation assumes the required keys are present (no defensive try/excepts).
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.weighted_same_per_degree = {deg: 0.0 for deg in range(0, 9)}
        self.weighted_same_per_degree_protein = {deg: [] for deg in range(0, 9)}
        self.sample_metrics = {'weighted_same_type_per_sample': [], 'cath_degree_per_sample': [], 'pair_infos': []}
        self.count_per_degree = {deg: 0 for deg in range(0, 9)}
        self.total_count = 0

    def update(self, batch, outputs):
        # No defensive guards: assume keys present
        batch_size = len(batch['metadata'])
        corr_values = outputs['corr_values']
        corr_atom_types = outputs['corr_atom_types']

        # compute the weighted mean of coroepsondnces with the same atom type. taking in account the masks
        for batch_id in range(batch_size):
            metadata = batch['metadata'][batch_id]
            cath_degree = metadata['cath_degree']
            pair_info = metadata.copy()
            for k in ['rotations', 'translations', 'rmse', 'coverage']:
                pair_info.pop(k, None)

            corr_vals_b = corr_values[batch_id]
            corr_types_b = corr_atom_types[batch_id]

            # determine same-type mask: corr_types_b may be Nx2 (types for each side)
            same_type = (corr_types_b[:, 0] == corr_types_b[:, 1]).float()

            weights = corr_vals_b
            weighted_same_val = (weights * same_type).sum()

            # record metrics
            self.count_per_degree[cath_degree] += 1
            self.sample_metrics['weighted_same_type_per_sample'].append(weighted_same_val)
            self.sample_metrics['cath_degree_per_sample'].append(cath_degree)
            self.sample_metrics['pair_infos'].append(pair_info)
            self.weighted_same_per_degree[cath_degree] += weighted_same_val
            self.weighted_same_per_degree_protein[cath_degree].append(weighted_same_val)
            self.total_count += 1

    def compute(self):
        weighted_same_per_degree_avg = {
            deg: (self.weighted_same_per_degree[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else torch.tensor(0.0) 
            for deg in range(0, 9)
        }

        overall = 0.0
        if self.total_count > 0:
            overall = sum(self.sample_metrics['weighted_same_type_per_sample']) / self.total_count

        return {
            'weighted_same_type_per_degree': weighted_same_per_degree_avg,
            'weighted_same_type_overall': overall,
            'weighted_same_type_per_degree_protein': self.weighted_same_per_degree_protein,
            **self.sample_metrics,
            'counts_per_degree': self.count_per_degree,
            'total_count': self.total_count
        }
