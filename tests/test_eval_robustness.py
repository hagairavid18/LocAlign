"""Evaluation and data-loading robustness: atom-type baseline, failed pairs, CDM index decoding.

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
from metrics.atom_type_correspondence import AtomTypeCorrespondence
from models.correspondences_denoiser import CDM
from models.utils.collate import custom_collate_fn


class AtomTypeBaselineTest(unittest.TestCase):
    def test_padding_is_not_counted_as_carbon(self):
        batch = {
            "metadata": [{"cath_degree": 4}],
            "src_atom_types": torch.tensor([[1, 2, 0, 0, 0, 0]]),
            "tar_atom_types": torch.tensor([[1, 2, 0, 0, 0, 0]]),
            "src_mask": torch.tensor([[True, True, False, False, False, False]]),
            "tar_mask": torch.tensor([[True, True, False, False, False, False]]),
        }
        outputs = {
            "corr_values": torch.tensor([[1.0]]),
            "corr_atom_types": torch.tensor([[[1, 1]]]),
        }
        metric = AtomTypeCorrespondence()
        metric.update(batch, outputs)
        weights = {1: 3.0, 2: 4.0}
        expected = sum(0.25 * w for w in weights.values()) / sum(0.5 * w for w in weights.values())
        self.assertAlmostEqual(metric.compute()["random_baseline_per_sample"][0], expected, places=6)


class FailedPairTest(unittest.TestCase):
    def test_collate_drops_failed_pairs(self):
        sample = {"x": torch.zeros(2), "metadata": {"pair_idx": 0}}
        batch = custom_collate_fn([None, sample, None])
        self.assertEqual(batch["x"].shape, (1, 2))
        self.assertEqual(batch["metadata"], [{"pair_idx": 0}])
        self.assertIsNone(custom_collate_fn([None, None]))

    def test_on_error_returns_none_without_resampling(self):
        holder = types.SimpleNamespace(_resample_on_error=False)
        self.assertIsNone(ScanNetDataset._on_error(holder))


class CDMIndexDecodingTest(unittest.TestCase):
    def test_non_square_matrix_decodes_to_tar_src_pairs(self):
        cdm = types.SimpleNamespace(_n_nodes=2)
        corr = torch.zeros(1, 2, 5)
        corr[0, 1, 4] = 3.0
        corr[0, 0, 3] = 2.0
        corr[0, 1, 0] = 1.0
        _, indices = CDM.extract_top_k_correspondences(cdm, corr)
        self.assertEqual(indices[0].tolist(), [[1, 4], [0, 3]])


if __name__ == "__main__":
    unittest.main()
