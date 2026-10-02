"""LigandLoss reports the ligand RMSD in Å, and the squashed value only as the loss term.

`models` is imported before `losses` to follow the package's import order (losses imports models).
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "aligner_dl"))

import models  # noqa: F401
from losses.soft_bb_loss import LigandLoss


def shifted_pair_batch(
    shift: tuple[float, float, float],
    n_atoms: int = 3,
    n_padding: int = 2,
) -> dict:
    """One pair whose target ligand is the source shifted by `shift`, plus masked padding atoms."""
    src = torch.rand(1, n_atoms + n_padding, 3)
    tar = src + torch.tensor(shift)
    tar[:, n_atoms:] += 100.0
    mask = torch.tensor([[True] * n_atoms + [False] * n_padding])
    return {"src_ligand_coordinates": src, "tar_ligand_coordinates": tar, "src_ligand_mask": mask}


class LigandRmsdTest(unittest.TestCase):
    def setUp(self):
        self.batch = shifted_pair_batch((3.0, 4.0, 0.0))
        self.rotation = torch.eye(3).unsqueeze(0)
        self.translation = torch.zeros(1, 3)
        self.loss = LigandLoss(return_non_linear=True)

    def test_raw_rmsd_is_in_angstrom(self):
        rmsd = self.loss(self.batch, self.rotation, self.translation, reduce=False, non_linear=False)
        self.assertAlmostEqual(rmsd.item(), 5.0, places=5)

    def test_squashed_value_is_only_the_loss_term(self):
        raw = self.loss(self.batch, self.rotation, self.translation, reduce=False, non_linear=False)
        squashed = self.loss(self.batch, self.rotation, self.translation, reduce=False)
        self.assertAlmostEqual(squashed.item(), 5.0 / (1 + 5.0 / 10.0), places=5)
        self.assertAlmostEqual(self.loss.transform(raw).item(), squashed.item(), places=6)


if __name__ == "__main__":
    unittest.main()
