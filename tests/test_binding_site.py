"""Residue-level binding site (ScanNetDataset._compute_pocket_mask) and BindingSiteCorrespondence.

`models` is imported before `datasets` and `metrics` to follow the package's import order.
"""
import os
import sys
import types
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "aligner_dl"))

import models  # noqa: F401
from datasets.scannet_dataset import ScanNetDataset
from metrics.binding_site_correspondence import BindingSiteCorrespondence


def pocket_mask(
    atom_coordinates: torch.Tensor,
    ligand_coordinates: torch.Tensor,
    atom_residue_indices: torch.Tensor,
) -> torch.Tensor:
    """Call _compute_pocket_mask without building a dataset."""
    holder = types.SimpleNamespace(_POCKET_DISTANCE_THRESHOLD=ScanNetDataset._POCKET_DISTANCE_THRESHOLD)
    return ScanNetDataset._compute_pocket_mask(holder, atom_coordinates, ligand_coordinates, atom_residue_indices)


class PocketMaskTest(unittest.TestCase):
    def test_whole_residue_is_in_the_pocket_when_one_atom_is_near(self):
        atoms = torch.tensor([[1.0, 0, 0], [9.0, 0, 0], [20.0, 0, 0], [21.0, 0, 0]])
        residues = torch.tensor([10, 10, 11, 11])
        ligand = torch.zeros(1, 3)
        self.assertEqual(pocket_mask(atoms, ligand, residues).tolist(), [True, True, False, False])

    def test_threshold_is_4_angstrom(self):
        atoms = torch.tensor([[4.0, 0, 0], [4.01, 0, 0]])
        residues = torch.tensor([1, 2])
        self.assertEqual(pocket_mask(atoms, torch.zeros(1, 3), residues).tolist(), [True, False])


class BindingSiteCorrespondenceTest(unittest.TestCase):
    def test_fraction_and_base_rate(self):
        batch = {
            "metadata": [{"cath_degree": 4}],
            "src_pocket_mask": torch.tensor([[True, False, False, False]]),
            "tar_pocket_mask": torch.tensor([[True, True, False, False]]),
            "src_mask": torch.tensor([[True, True, True, True]]),
            "tar_mask": torch.tensor([[True, True, True, False]]),
        }
        outputs = {
            "corr_values": torch.tensor([[3.0, 1.0]]),
            "corr_pocket_mask": torch.tensor([[[True, True], [False, True]]]),
        }
        metric = BindingSiteCorrespondence()
        metric.update(batch, outputs)
        result = metric.compute()
        self.assertAlmostEqual(result["weighted_pocket_fraction_overall"], 0.75)
        self.assertAlmostEqual(result["pocket_base_rate_per_sample"][0], 0.5 * (1 / 4 + 2 / 3))


if __name__ == "__main__":
    unittest.main()
