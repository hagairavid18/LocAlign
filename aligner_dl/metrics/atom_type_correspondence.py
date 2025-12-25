import logging
from torch.nn import Module
import torch

logger = logging.getLogger(__name__)

# constant per-atom-type importance weights: 0:C, 1:O, 2:N, 3:S
ATOM_TYPE_WEIGHTS = torch.tensor([1.0, 3.0, 4.0, 50.0], dtype=torch.float32)


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
        self.random_baseline_per_degree = {deg: 0.0 for deg in range(0, 9)}
        self.sample_metrics = {'weighted_same_type_per_sample': [], 'cath_degree_per_sample': [], 'pair_infos': [], 'random_baseline_per_sample': []}
        self.count_per_degree = {deg: 0 for deg in range(0, 9)}
        self.total_count = 0

    def update(self, batch, outputs):
        batch_size = len(batch['metadata'])
        corr_values = outputs['corr_values']
        corr_atom_types = outputs['corr_atom_types']
        
        # We also need the raw atom types of the source and target to calculate 
        # the true random expectation (not just the types in the correspondences)
        src_atom_types_all = batch['src_atom_types'] # Shape: [B, N]
        tar_atom_types_all = batch['tar_atom_types'] # Shape: [B, M]
        tw = ATOM_TYPE_WEIGHTS.to(corr_atom_types.device)

        for batch_id in range(batch_size):
            metadata = batch['metadata'][batch_id]
            cath_degree = metadata['cath_degree']
            
            # --- 1. Compute Actual Weighted Score (Your existing logic) ---
            corr_vals_b = corr_values[batch_id].float()
            corr_types_b = corr_atom_types[batch_id]
            
            same_type = (corr_types_b[:, 0] == corr_types_b[:, 1]).float()
            src_types_corr = corr_types_b[:, 0].long()
            tar_types_corr = corr_types_b[:, 1].long()
            
            wt_src = tw[src_types_corr.clamp(max=len(tw)-1)]
            wt_tar = tw[tar_types_corr.clamp(max=len(tw)-1)]
            pair_weight = (wt_src + wt_tar) * 0.5
            
            weights_eff = corr_vals_b * pair_weight
            actual_weighted_val = (weights_eff * same_type).sum() / weights_eff.sum()

            # --- 2. Compute Random Expectation Baseline ---
            # Get distribution of types in the full protein structures
            s_types = src_atom_types_all[batch_id]
            t_types = tar_atom_types_all[batch_id]
            
            # Calculate P(type) for src and tar
            num_types = len(tw)
            p_s = torch.bincount(s_types, minlength=num_types).float() / len(s_types)
            p_t = torch.bincount(t_types, minlength=num_types).float() / len(t_types)
            
            # Expected numerator: Sum over i [ P_s(i) * P_t(i) * Weight(i) ]
            expected_num = torch.sum(p_s * p_t * tw)
            
            # Expected denominator: Average of the mean weights
            mean_wt_s = torch.sum(p_s * tw)
            mean_wt_t = torch.sum(p_t * tw)
            expected_den = 0.5 * (mean_wt_s + mean_wt_t)
            
            random_expected_val = (expected_num / expected_den).item()

            # --- 3. Store Metrics ---
            self.weighted_same_per_degree[cath_degree] += actual_weighted_val.item()
            self.random_baseline_per_degree[cath_degree] += random_expected_val
            self.count_per_degree[cath_degree] += 1
            # You can now track the gap: (Actual - Random)
            self.sample_metrics['weighted_same_type_per_sample'].append(actual_weighted_val.item())
            self.sample_metrics['random_baseline_per_sample'].append(random_expected_val)
            # Track per-protein/per-degree values for downstream analysis
            self.weighted_same_per_degree_protein[cath_degree].append(actual_weighted_val.item())
            
            self.total_count += 1

    def compute(self):
        weighted_same_per_degree_avg = {
            deg: (self.weighted_same_per_degree[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else 0.0
            for deg in range(0, 9)
        }
        random_baseline_per_degree_avg = {
            deg: (self.random_baseline_per_degree[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else 0.0
            for deg in range(0, 9)
        }

        overall = 0.0
        random_baseline_overall = 0.0
        if self.total_count > 0:
            overall = sum(self.sample_metrics['weighted_same_type_per_sample']) / self.total_count
            random_baseline_overall = sum(self.sample_metrics['random_baseline_per_sample']) / self.total_count

        return {
            'weighted_same_type_per_degree': weighted_same_per_degree_avg,
            'random_baseline_per_degree': random_baseline_per_degree_avg,
            'weighted_same_type_overall': overall,
            'random_baseline_overall': random_baseline_overall,
            'weighted_same_type_per_degree_protein': self.weighted_same_per_degree_protein,
            **self.sample_metrics,
            'counts_per_degree': self.count_per_degree,
            'total_count': self.total_count
        }
