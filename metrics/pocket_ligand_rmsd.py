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
        self.total = 0
        self.count = 0
        self.pocket_rmsd = 0
        self.ligand_rmsd = 0
        self.pocket_rmsd_first_iter = 0
        self.ligand_rmsd_first_iter = 0

    def update(self, batch, outputs):
        batch_size = batch['tar_embedding'].shape[0]
        for batch_id in range(batch_size):
            ligand_res_idx = 0
            mov_protein = Protein(batch['metadata'][batch_id]['mov_protein'], batch['metadata'][batch_id]['mov_chain'], batch['metadata'][batch_id]['Ligand_ID'], save_models=False)
            batch['metadata'][0]['ref_protein']
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
            self.pocket_rmsd += pocket_rmsd
            self.ligand_rmsd += ligand_rmsd
            if 'pred_first_R' in outputs:
                pred_T[:3, :3] = outputs['pred_first_R'][batch_id]
                pred_T[:3, 3] = outputs['pred_first_t'][batch_id]
                pocket_rmsd_first = ProteinPair.compute_rmsd(pocket_atoms, gt_T.cpu().numpy(), pred_T.cpu().numpy())
                ligand_rmsd_first = ProteinPair.compute_rmsd(np.vstack(ligand_atoms_coors), gt_T.cpu().numpy(), pred_T.cpu().numpy())
                self.pocket_rmsd_first_iter += pocket_rmsd_first
                self.ligand_rmsd_first_iter += ligand_rmsd_first

            
        self.total += batch_size
    def compute(self):
        return {'pocket_rmsd': self.pocket_rmsd / self.total, 'ligand_rmsd': self.ligand_rmsd / self.total, 'pocket_rmsd_iter0': self.pocket_rmsd_first_iter / self.total, 'ligand_rmsd_iter0': self.ligand_rmsd_first_iter / self.total}
