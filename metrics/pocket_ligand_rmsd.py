import numpy as np
import logging

import torch

from aligners import *
from objects import ProteinPair, Protein
from torch.nn import Module


logger = logging.getLogger(__name__)

class PocketRMSD(Module):
    def __init__(self):
        super(PocketRMSD, self).__init__()
        self.reset()

    def reset(self):
        # Initialize metrics for each degree 1 through 8
        self.pocket_rmsd_per_degree = {deg: 0 for deg in range(1, 9)}
        self.ligand_rmsd_per_degree = {deg: 0 for deg in range(1, 9)}
        self.pocket_rmsd_first_iter_per_degree = {deg: 0 for deg in range(1, 9)}
        self.ligand_rmsd_first_iter_per_degree = {deg: 0 for deg in range(1, 9)}
        self.count_per_degree = {deg: 0 for deg in range(1, 9)}
        self.total_count = 0
        self.sample_metrics = {'cath_degree_per_sample': [], 'pocket_rmsd_per_sample': []}

    def update(self, batch, outputs):
        batch_size = batch['tar_embedding'].shape[0]
        for batch_id in range(batch_size):
            cath_degree = batch['metadata'][batch_id]['cath_degree']
            ligand_res_idx = 0
            mov_protein = Protein(batch['metadata'][batch_id]['mov_protein'], batch['metadata'][batch_id]['mov_chain'], batch['metadata'][batch_id]['Ligand_ID'], save_models=False)
            # batch['metadata'][0]['ref_protein']
            pocket_atoms = mov_protein.get_pocket_atoms(ligand_res_idx = ligand_res_idx) # TODO
            gt_T, pred_T = torch.eye(4), torch.eye(4)
            gt_T[:3, :3] = batch['gt_R'][batch_id]
            gt_T[:, 3] = batch['gt_t'][batch_id]
            pred_T[:3, :3] = outputs['pred_R'][batch_id]
            pred_T[:3, 3] = outputs['pred_t'][batch_id]
            pocket_rmsd = ProteinPair.compute_rmsd(pocket_atoms, gt_T.cpu().numpy(), pred_T.cpu().numpy())

            ligand_residue = mov_protein.get_ligand_residues()[ligand_res_idx] # TODO: handle ligand with more residues
            ligand_atoms_coors = [atom.coord for atom in ligand_residue.get_atoms() if atom.element != "H"]
            ligand_rmsd = ProteinPair.compute_rmsd(np.vstack(ligand_atoms_coors), gt_T.cpu().numpy(), pred_T.cpu().numpy())
           
            self.pocket_rmsd_per_degree[cath_degree] += pocket_rmsd
            self.ligand_rmsd_per_degree[cath_degree] += ligand_rmsd
            self.count_per_degree[cath_degree] += 1
            self.sample_metrics['cath_degree_per_sample'].append(cath_degree)
            self.sample_metrics['pocket_rmsd_per_sample'].append(pocket_rmsd)
            if 'pred_first_R' in outputs:
                pred_T[:3, :3] = outputs['pred_first_R'][batch_id]
                pred_T[:3, 3] = outputs['pred_first_t'][batch_id]
                pocket_rmsd_first = ProteinPair.compute_rmsd(pocket_atoms, gt_T.cpu().numpy(), pred_T.cpu().numpy())
                ligand_rmsd_first = ProteinPair.compute_rmsd(np.vstack(ligand_atoms_coors), gt_T.cpu().numpy(), pred_T.cpu().numpy())
                self.pocket_rmsd_first_iter_per_degree[cath_degree] += pocket_rmsd_first
                self.ligand_rmsd_first_iter_per_degree[cath_degree] += ligand_rmsd_first

                
        self.total_count += batch_size

    def compute(self):
        # Calculate overall averages by summing all per-degree values and dividing by total count
        total_metrics = {
            'pocket_rmsd': sum(self.pocket_rmsd_per_degree.values()) / self.total_count if self.total_count > 0 else 0,
            'ligand_rmsd': sum(self.ligand_rmsd_per_degree.values()) / self.total_count if self.total_count > 0 else 0,
            'pocket_rmsd_iter0': sum(self.pocket_rmsd_first_iter_per_degree.values()) / self.total_count if self.total_count > 0 else 0,
            'ligand_rmsd_iter0': sum(self.ligand_rmsd_first_iter_per_degree.values()) / self.total_count if self.total_count > 0 else 0
        }

        # Calculate per-degree averages, handling zero counts
        per_degree_metrics = {
            'pocket_rmsd': {deg: (self.pocket_rmsd_per_degree[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else 0 for deg in range(1, 9)},
            'ligand_rmsd': {deg: (self.ligand_rmsd_per_degree[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else 0 for deg in range(1, 9)},
            'pocket_rmsd_iter0': {deg: (self.pocket_rmsd_first_iter_per_degree[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else 0 for deg in range(1, 9)},
            'ligand_rmsd_iter0': {deg: (self.ligand_rmsd_first_iter_per_degree[deg] / self.count_per_degree[deg]) if self.count_per_degree[deg] > 0 else 0 for deg in range(1, 9)}
        }
        counts = {
            'counts_per_degree': self.count_per_degree,
            'total_count': self.total_count
        }

        # Return both total and per-degree metrics
        return {**total_metrics, **per_degree_metrics, **counts, **self.sample_metrics}