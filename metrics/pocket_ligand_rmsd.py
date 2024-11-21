import os
import pickle
import numpy as np
import logging

import torch

from aligners import *
from objects import ProteinPair, Protein
from torch.nn import Module

from utils.constants import LIGAND_DIR


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
        batch_size = len(batch['metadata'])
        for batch_id in range(batch_size):
            metadata = batch['metadata'][batch_id]
            cath_degree = metadata['cath_degree']
            ligand_res_idx = 0


            # Load or compute ligand residues
            ligand_residue = self._load_or_compute_ligand_residues(
                mov_protein=metadata['mov_protein'],
                mov_chain=metadata['mov_chain'],
                ligand_id=metadata['Ligand_ID'],
                ligand_res_idx=ligand_res_idx
            )

            # Extract ligand atom coordinates
            ligand_atoms_coors = [atom.coord for atom in ligand_residue.get_atoms() if atom.element != "H"]

            # Compute transformations
            gt_T, pred_T = torch.eye(4), torch.eye(4)
            gt_T[:3, :3] = batch['gt_R'][batch_id]
            gt_T[:, 3] = batch['gt_t'][batch_id]
            pred_T[:3, :3] = outputs['pred_R'][batch_id]
            pred_T[:3, 3] = outputs['pred_t'][batch_id]

            # Compute RMSDs
            pocket_rmsd = ProteinPair.compute_rmsd_torch(batch['src_pocket'], batch['gt_R'], batch['gt_t'], outputs['pred_R'], outputs['pred_t'], batch['src_pocket_mask'])[batch_id]
            ligand_rmsd = ProteinPair.compute_rmsd(np.vstack(ligand_atoms_coors), gt_T.cpu().numpy(), pred_T.cpu().numpy())

            # Update metrics
            self.pocket_rmsd_per_degree[cath_degree] += pocket_rmsd
            self.ligand_rmsd_per_degree[cath_degree] += ligand_rmsd
            self.count_per_degree[cath_degree] += 1
            self.sample_metrics['cath_degree_per_sample'].append(cath_degree)
            self.sample_metrics['pocket_rmsd_per_sample'].append(pocket_rmsd.cpu())

            # First iteration metrics
            if 'pred_first_R' in outputs:
                pred_T[:3, :3] = outputs['pred_first_R'][batch_id]
                pred_T[:3, 3] = outputs['pred_first_t'][batch_id]
                pocket_rmsd_first = ProteinPair.compute_rmsd_torch(batch['src_pocket'], batch['gt_R'], batch['gt_t'], outputs['pred_first_R'], outputs['pred_first_t'], batch['src_pocket_mask'])[batch_id]
                ligand_rmsd_first = ProteinPair.compute_rmsd(np.vstack(ligand_atoms_coors), gt_T.cpu().numpy(), pred_T.cpu().numpy())
                self.pocket_rmsd_first_iter_per_degree[cath_degree] += pocket_rmsd_first
                self.ligand_rmsd_first_iter_per_degree[cath_degree] += ligand_rmsd_first
        self.total_count += batch_size


    # Helper functions
    def _load_or_compute_pocket_atoms(self, mov_protein: str, mov_chain: str, ligand_id: str, ligand_res_idx: int) -> np.ndarray:
        cache_dir = f"{LIGAND_DIR}/{ligand_id}/cache"
        pocket_atoms_file = os.path.join(cache_dir, f"{mov_protein}_pocket_atoms.pkl")
        if os.path.exists(pocket_atoms_file):
            with open(pocket_atoms_file, 'rb') as f:
                return pickle.load(f)
        else:
            mov_protein_obj = Protein(mov_protein, mov_chain, ligand_id, save_models=False)
            pocket_atoms = mov_protein_obj.get_pocket_atoms(ligand_res_idx=ligand_res_idx)
            os.makedirs(cache_dir, exist_ok=True)
            with open(pocket_atoms_file, 'wb') as f:
                pickle.dump(pocket_atoms, f)
            return pocket_atoms

    def _load_or_compute_ligand_residues(self, mov_protein: str, mov_chain: str, ligand_id: str, ligand_res_idx: int):
        cache_dir = f"{LIGAND_DIR}/{ligand_id}/cache"
        ligand_residues_file = os.path.join(cache_dir, f"{mov_protein}_ligand_residues.pkl")
        if os.path.exists(ligand_residues_file):
            with open(ligand_residues_file, 'rb') as f:
                ligand_residues = pickle.load(f)
        else:
            mov_protein_obj = Protein(mov_protein, mov_chain, ligand_id, save_models=False)
            ligand_residues = mov_protein_obj.get_ligand_residues()
            os.makedirs(cache_dir, exist_ok=True)
            with open(ligand_residues_file, 'wb') as f:
                pickle.dump(ligand_residues, f)
        return ligand_residues[ligand_res_idx]


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